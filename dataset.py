import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta

class PVDataset(Dataset):
    """
    Physics-Informed Dataset for Multi-site PV Generation.
    
    Data Structure expected in input DataFrame:
    - timestamp_win: Base timestamp for each sequence (15min resolution).
    - observe_power: np.ndarray [672] (History 7 days)
    - observe_power_future: np.ndarray [192] (Future 2 days)
    - GHI_solargis: np.ndarray [672] (History 7 days NWP GHI)
    - GHI_solargis_future: np.ndarray [192] (Future 2 days NWP GHI)
    - TEMP_solargis: np.ndarray [672] (History 7 days NWP Temperature)
    - TEMP_solargis_future: np.ndarray [192] (Future 2 days NWP Temperature)
    - station: Station ID
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

    def _compute_clearsky_ghi(self, start_time: pd.Timestamp, lat: float, lon: float, tz: str) -> np.ndarray:
        """
        Compute clearsky GHI for the target 1-day sequence.
        """
        loc = Location(lat, lon, tz=tz)
        times = pd.date_range(
            start=start_time, 
            periods=self.future_len, 
            freq='15min', 
            tz=tz
        )
        # Use Ineichen model for clearsky
        clearsky = loc.get_clearsky(times)
        return clearsky['ghi'].values # [future_len]

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
        
        return np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=-1) # [future_len, 4]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use iterative retry instead of recursion to avoid stack overflow
        max_retries = 10

        for _ in range(max_retries):
            row = self.df.iloc[idx]
            station_id = row['station']
            meta = self.station_metadata[station_id]

            # Parse timestamp (the start of future window)
            start_time = pd.to_datetime(row['timestamp_win'])
            if start_time.tzinfo is None:
                start_time = start_time.tz_localize('UTC').tz_convert(meta['tz'])
            else:
                start_time = start_time.tz_convert(meta['tz'])

            # Only use the first 96 steps (1 day) of the future data
            # Power [future_len, 1]
            power = row['observe_power_future'][:self.future_len]

            # NWP features: [future_len, 2] (GHI, TEMP)
            nwp_ghi = row['GHI_solargis_future'][:self.future_len]
            nwp_temp = row['TEMP_solargis_future'][:self.future_len]
            nwp_base = np.stack([nwp_ghi, nwp_temp], axis=-1)

            # Physics Injection: Clear-sky GHI for 1 day
            ghi_clearsky = self._compute_clearsky_ghi(start_time, meta['lat'], meta['lon'], meta['tz'])

            # --- Anomaly Filtering ---
            high_ghi = (ghi_clearsky > 500)
            is_zero_power = (power == 0)

            if high_ghi.sum() > 0 and (is_zero_power & high_ghi).sum() > 0.1 * high_ghi.sum():
                idx = np.random.randint(0, len(self))
                continue

            break
        
        # Time Features: [future_len, 4]
        time_feats = self._get_time_features(start_time, meta['tz'])
        
        # Final NWP feature tensor: Base NWP + Clear-sky GHI + Time Features
        # [96, 2 + 1 + 4] = [96, 7]
        nwp_full = np.concatenate([nwp_base, ghi_clearsky[:, np.newaxis], time_feats], axis=-1)
        
        return {
            'nwp': torch.tensor(nwp_full, dtype=torch.float32),      # [96, 7]
            'ghi_clearsky': torch.tensor(ghi_clearsky, dtype=torch.float32).unsqueeze(-1), # [96, 1]
            'power': torch.tensor(power, dtype=torch.float32).unsqueeze(-1), # [96, 1]
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
