import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta

def compute_clearsky_series(
    df: pd.DataFrame, 
    station_metadata: Dict[Any, Dict[str, float]], 
    future_len: int = 192
) -> np.ndarray:
    """
    Standalone interface to compute clearsky GHI for a dataset.
    Can be used for pre-processing.
    """
    clearsky_ghi_all = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        station_id = row['station']
        meta = station_metadata[station_id]
        
        # Parse and localize timestamp
        start_time = pd.to_datetime(row['timestamp_win'])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC').tz_convert(meta['tz'])
        else:
            start_time = start_time.tz_convert(meta['tz'])
            
        loc = Location(meta['lat'], meta['lon'], tz=meta['tz'])
        times = pd.date_range(
            start=start_time, 
            periods=future_len, 
            freq='15min', 
            tz=meta['tz']
        )
        clearsky = loc.get_clearsky(times)
        clearsky_ghi_all.append(clearsky['ghi'].values.astype(np.float32))
    
    return np.array(clearsky_ghi_all)

def compute_time_features_series(
    df: pd.DataFrame, 
    station_metadata: Dict[Any, Dict[str, float]], 
    future_len: int = 192
) -> np.ndarray:
    """
    Standalone interface to compute time features (Sin/Cos) for a dataset.
    """
    time_feats_all = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        station_id = row['station']
        meta = station_metadata[station_id]
        
        # Parse and localize timestamp
        start_time = pd.to_datetime(row['timestamp_win'])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC').tz_convert(meta['tz'])
        else:
            start_time = start_time.tz_convert(meta['tz'])
            
        times = pd.date_range(
            start=start_time, 
            periods=future_len, 
            freq='15min', 
            tz=meta['tz']
        )
        
        hours = times.hour + times.minute / 60.0
        months = times.month - 1 # 0-11
        
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        month_sin = np.sin(2 * np.pi * months / 12.0)
        month_cos = np.cos(2 * np.pi * months / 12.0)
        
        time_feats_all.append(np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=-1).astype(np.float32))
    
    return np.array(time_feats_all)

class PVDataset(Dataset):
    """
    Physics-Informed Dataset for Multi-site PV Generation.
    Supports Stage 1 (Contrastive Fingerprinting) and Stage 2 (Conditional Generation).
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        station_metadata: Dict[Any, Dict[str, float]], 
        context_len: int = 96 * 14, 
        future_len: int = 96
    ):
        self.df = df
        self.station_metadata = station_metadata
        self.context_len = context_len
        self.future_len = future_len
        
        # Group indices by station to facilitate same-site sampling
        self.station_groups = df.groupby('station').indices
        self.station_list = list(self.station_groups.keys())
        
        # Pre-compute clearsky GHI and time features for all samples (Target Window only)
        # In a real system, we'd pre-compute or compute on-the-fly.
        # For simplicity, we assume the input df has 'GHI_solargis_future' for the 1-day target.
        print(f"Pre-computing features for {len(df)} samples...")
        self.clearsky_ghi_all = compute_clearsky_series(df, station_metadata, future_len)
        self.time_feats_all = compute_time_features_series(df, station_metadata, future_len)

    def __len__(self) -> int:
        return len(self.df)

    def _get_single_view(self, idx: int, is_context: bool = True) -> torch.Tensor:
        """Helper to construct the [Seq, 8] feature tensor for a specific window."""
        row = self.df.iloc[idx]
        
        if is_context:
            # Context window (e.g. 14 days)
            # For simplicity, we assume GHI/TEMP/Power are available for context_len
            ghi = row['GHI_solargis'][:self.context_len]
            temp = row['TEMP_solargis'][:self.context_len]
            power = row['observe_power'][:self.context_len]
            
            # Pad with zeros for missing NWP features (Hour_Sin etc.) if not available for context
            # In production, we'd compute full NWP for context too.
            # Here we just use [GHI, Temp, Power, 0, 0, 0, 0, 0] to match input_dim=8
            feat = np.zeros((self.context_len, 8), dtype=np.float32)
            feat[:, 0] = ghi
            feat[:, 1] = temp
            feat[:, 2] = power
            return torch.tensor(feat, dtype=torch.float32)
        else:
            # Target window (1 day) - Full 7 NWP features
            ghi_cs = self.clearsky_ghi_all[idx]
            time_f = self.time_feats_all[idx]
            nwp_ghi = row['GHI_solargis_future'][:self.future_len]
            nwp_temp = row['TEMP_solargis_future'][:self.future_len]
            
            # Base NWP (GHI, TEMP) + ClearSky + TimeFeats (4) = 7
            nwp_full = np.concatenate([
                nwp_ghi[:, None], 
                nwp_temp[:, None], 
                ghi_cs[:, None], 
                time_f
            ], axis=-1)
            return torch.tensor(nwp_full, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        station_id = row['station']
        
        # View 1: Context Anchor (History at time T1)
        context_anchor = self._get_single_view(idx, is_context=True)
        
        # View 2: Context Positive (Another time T2 for same station)
        station_indices = self.station_groups[station_id]
        if len(station_indices) > 1:
            pos_idx = np.random.choice(station_indices)
            # Ensure it's a different time window if possible
            while pos_idx == idx and len(station_indices) > 5:
                pos_idx = np.random.choice(station_indices)
        else:
            pos_idx = idx
        context_pos = self._get_single_view(pos_idx, is_context=True)
        
        # Target Data for generation (at time T1)
        target_nwp = self._get_single_view(idx, is_context=False)
        target_power = torch.tensor(row['observe_power_future'][:self.future_len], dtype=torch.float32).unsqueeze(-1)
        ghi_cs = torch.tensor(self.clearsky_ghi_all[idx], dtype=torch.float32).unsqueeze(-1)
        
        return {
            'context_anchor': context_anchor, # [Context_Len, 8]
            'context_pos': context_pos,       # [Context_Len, 8]
            'target_nwp': target_nwp,         # [Future_Len, 7]
            'target_power': target_power,     # [Future_Len, 1]
            'ghi_clearsky': ghi_cs,           # [Future_Len, 1]
            'site_id': torch.tensor(int(station_id), dtype=torch.long)
        }

if __name__ == "__main__":
    # Test with dummy data
    history_len = 672
    future_len = 192
    
    dummy_data = {
        'timestamp_win': [pd.Timestamp('2023-01-08 00:00:00')],
        'observe_power': [np.random.rand(history_len)],
        'observe_power_future': [np.random.rand(future_len)],
        'GHI_solargis': [np.random.rand(history_len) * 1000],
        'GHI_solargis_future': [np.random.rand(future_len) * 1000],
        'TEMP_solargis': [np.random.rand(history_len) * 30],
        'TEMP_solargis_future': [np.random.rand(future_len) * 30],
        'station': [1]
    }
    df = pd.DataFrame(dummy_data)
    
    meta = {1: {'lat': 31.2304, 'lon': 121.4737, 'tz': 'Asia/Shanghai'}} # Shanghai
    
    ds = PVDataset(df, meta)
    sample = ds[0]
    
    print(f"NWP Shape: {sample['nwp'].shape}")
    print(f"GHI Clearsky Shape: {sample['ghi_clearsky'].shape}")
    print(f"Power Shape: {sample['power'].shape}")
    print(f"Site ID: {sample['site_id']}")
