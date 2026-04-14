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

class PVDataset(Dataset):
    """
    Physics-Informed Dataset for Multi-site PV Generation.
    
    Data Structure expected in input DataFrame:
    - timestamp_win: Base timestamp for each sequence (15min resolution).
    - observe_power_future: np.ndarray [future_len] (Future power)
    - GHI_solargis_future: np.ndarray [future_len] (Future NWP GHI)
    - TEMP_solargis_future: np.ndarray [future_len] (Future NWP Temperature)
    - station: Station ID
    - ghi_clearsky: Optional[np.ndarray] [future_len] (Pre-computed clear-sky GHI)
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        station_metadata: Dict[Any, Dict[str, float]], 
        history_len: int = 672, 
        future_len: int = 192
    ):
        """
        Args:
            df: DataFrame containing the PV data.
            station_metadata: Dict mapping station_id to {'lat': float, 'lon': float, 'tz': str}.
            history_len: Length of historical sequence (default 7 days @ 15min).
            future_len: Length of future sequence (default 2 days @ 15min).
        """
        self.df = df
        self.station_metadata = station_metadata
        self.history_len = history_len
        self.future_len = future_len
        self.total_len = history_len + future_len
        
        # 1. Handle Clearsky GHI (Check if pre-computed in DF)
        if 'ghi_clearsky' in self.df.columns:
            print("Using pre-computed clearsky GHI from DataFrame.")
            self.clearsky_ghi_all = np.stack(self.df['ghi_clearsky'].values)
        else:
            print(f"Pre-computing clearsky GHI for {len(df)} samples...")
            self.clearsky_ghi_all = compute_clearsky_series(df, station_metadata, future_len)
        
        # 2. Pre-compute time features
        print(f"Pre-computing time features for {len(df)} samples...")
        self.time_feats_all = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            station_id = row['station']
            meta = self.station_metadata[station_id]
            start_time = pd.to_datetime(row['timestamp_win'])
            if start_time.tzinfo is None:
                start_time = start_time.tz_localize('UTC').tz_convert(meta['tz'])
            else:
                start_time = start_time.tz_convert(meta['tz'])
            
            time_f = self._get_time_features(start_time, meta['tz'])
            self.time_feats_all.append(time_f)
            
        self.time_feats_all = np.array(self.time_feats_all)

    def _get_time_features(self, start_time: pd.Timestamp, tz: str) -> np.ndarray:
        """
        Compute Sin/Cos positional encoding for hour-of-day and month-of-year.
        """
        times = pd.date_range(
            start=start_time, 
            periods=self.future_len, 
            freq='15min', 
            tz=tz
        )
        
        hours = times.hour + times.minute / 60.0
        months = times.month - 1 # 0-11
        
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        month_sin = np.sin(2 * np.pi * months / 12.0)
        month_cos = np.cos(2 * np.pi * months / 12.0)
        
        return np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=-1).astype(np.float32) # [future_len, 4]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use iterative retry instead of recursion to avoid stack overflow
        max_retries = 10

        for _ in range(max_retries):
            row = self.df.iloc[idx]
            station_id = row['station']
            
            # Use pre-computed clearsky GHI and time features
            ghi_clearsky = self.clearsky_ghi_all[idx]
            time_feats = self.time_feats_all[idx]

            # Only use the first future_len steps of the future data
            power = row['observe_power_future'][:self.future_len]

            # NWP features: [future_len, 2] (GHI, TEMP)
            nwp_ghi = row['GHI_solargis_future'][:self.future_len]
            nwp_temp = row['TEMP_solargis_future'][:self.future_len]
            nwp_base = np.stack([nwp_ghi, nwp_temp], axis=-1)

            # --- Anomaly Filtering ---
            high_ghi = (ghi_clearsky > 500)
            is_zero_power = (power == 0)

            if high_ghi.sum() > 0 and (is_zero_power & high_ghi).sum() > 0.1 * high_ghi.sum():
                idx = np.random.randint(0, len(self))
                continue

            break
        
        # Final NWP feature tensor: Base NWP + Clear-sky GHI + Time Features
        nwp_full = np.concatenate([nwp_base, ghi_clearsky[:, np.newaxis], time_feats], axis=-1)
        
        return {
            'nwp': torch.tensor(nwp_full, dtype=torch.float32),
            'ghi_clearsky': torch.tensor(ghi_clearsky, dtype=torch.float32).unsqueeze(-1),
            'power': torch.tensor(power, dtype=torch.float32).unsqueeze(-1),
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
