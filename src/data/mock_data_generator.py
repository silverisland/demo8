import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import PVDataset
from ..utils.config import config

def generate_mock_dataloader(num_samples=200, batch_size=16):
    """
    Generates a mock DataLoader with:
    - 14-day history (Context)
    - 1-day future (Target)
    - Multiple samples per station for same-site contrastive sampling.
    """
    context_len = config.context_len
    future_len = config.future_len
    
    dummy_data = []
    # Use 10 unique stations
    for station_id in range(1, 11):
        # Generate 20 samples per station at different time windows
        for i in range(20):
            start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(15, 300))
            
            def get_ghi(length):
                t = np.arange(length)
                diurnal = np.maximum(0, np.sin(np.pi * (t % 96 - 24) / 48))
                ghi = 1000 * diurnal + np.random.normal(0, 50, length)
                return np.maximum(0, ghi)

            # Total length: context_len + future_len
            ghi_total = get_ghi(context_len + future_len)
            
            def get_power(ghi):
                efficiency = 0.15 + np.random.normal(0, 0.02)
                power = efficiency * ghi
                num_clouds = int(len(ghi) / 10)
                for _ in range(num_clouds):
                    idx = np.random.randint(0, len(power))
                    width = np.random.randint(2, 10)
                    power[idx:idx+width] *= np.random.uniform(0.1, 0.6)
                return np.maximum(0, power)
            
            power_total = get_power(ghi_total)
            
            dummy_data.append({
                'timestamp_win': start_time,
                'observe_power': power_total[:context_len],
                'observe_power_future': power_total[context_len:],
                'GHI_solargis': ghi_total[:context_len],
                'GHI_solargis_future': ghi_total[context_len:],
                'TEMP_solargis': np.random.normal(25, 5, context_len),
                'TEMP_solargis_future': np.random.normal(25, 5, future_len),
                'station': station_id
            })
    
    df = pd.DataFrame(dummy_data)
    station_metadata = {i: {'lat': 31.23, 'lon': 121.47, 'tz': 'Asia/Shanghai'} for i in range(1, 11)}
    
    dataset = PVDataset(df, station_metadata, context_len, future_len)
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
