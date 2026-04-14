import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import PVDataset
from ..utils.config import config

def generate_mock_dataloader(num_samples=100, batch_size=16):
    """
    Generates a mock DataLoader with 1-day (96 steps) rugged PV power data.
    """
    future_len = config.future_len
    
    # 1. Create Mock DataFrame
    dummy_data = []
    for i in range(num_samples):
        # Random station ID (1 to 5)
        station_id = np.random.randint(1, 6)
        
        # Start time (random date in 2023)
        start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 300))
        
        # GHI: Rough bell shape for 1 day
        def get_ghi(length):
            t = np.arange(length)
            # Center at midday (48 steps)
            diurnal = np.maximum(0, np.sin(np.pi * (t - 24) / 48))
            ghi = 1000 * diurnal + np.random.normal(0, 50, length)
            return np.maximum(0, ghi)

        ghi_fut = get_ghi(future_len)
        
        # Power: Derived from GHI with "rugged" cloud effects
        def get_power(ghi):
            efficiency = 0.15 + np.random.normal(0, 0.02)
            power = efficiency * ghi
            # Cloud drops for 1 day (fewer clouds than 7 days)
            num_clouds = int(len(ghi) / 10)
            for _ in range(num_clouds):
                idx = np.random.randint(0, len(power))
                width = np.random.randint(2, 6)
                power[idx:idx+width] *= np.random.uniform(0.1, 0.6)
            power += np.random.normal(0, 2, len(power))
            return np.maximum(0, power)
        
        power_fut = get_power(ghi_fut)
        
        dummy_data.append({
            'timestamp_win': start_time,
            # Placeholder for history (not used)
            'observe_power': np.zeros(0), 
            'observe_power_future': power_fut,
            'GHI_solargis': np.zeros(0),
            'GHI_solargis_future': ghi_fut,
            'TEMP_solargis': np.zeros(0),
            'TEMP_solargis_future': np.random.normal(25, 5, future_len),
            'station': station_id
        })
    
    df = pd.DataFrame(dummy_data)
    
    # 2. Mock Station Metadata
    station_metadata = {
        i: {'lat': 31.23, 'lon': 121.47, 'tz': 'Asia/Shanghai'} 
        for i in range(1, 6)
    }
    
    # 3. Create Dataset and DataLoader
    dataset = PVDataset(df, station_metadata, history_len, future_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

if __name__ == "__main__":
    loader = generate_mock_dataloader(num_samples=20, batch_size=4)
    batch = next(iter(loader))
    print("Mock Data Generated Successfully!")
    print(f"NWP batch shape: {batch['nwp'].shape}") # [B, 864, 7]
    print(f"Power batch shape: {batch['power'].shape}") # [B, 864, 1]
    print(f"GHI Clearsky batch shape: {batch['ghi_clearsky'].shape}") # [B, 864, 1]
    print(f"Site ID batch shape: {batch['site_id'].shape}") # [B]
