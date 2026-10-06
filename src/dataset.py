import os
import torch
import numpy as np
import lightkurve as lk
from torch.utils.data import Dataset

class TESSTensorDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.file_list = []
        
        # Recursively search all nested subdirectories created by MAST/lightkurve
        for root, _, files in os.walk(data_dir):
            for file in files:
                if (file.endswith("-tp.fits") or file.endswith("-tp.fits.gz") or file.endswith(".fits") or file.endswith(".fits.gz")) and not file.startswith("."):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 1. Load the TPF from the local cached file
        tpf = lk.TessTargetPixelFile(file_path, quality_bitmask='hard')
        
        # 2. Extract the 3D flux array: (Time, Height, Width)
        flux = tpf.flux.value
        
        # 3. Clean NaNs caused by instrumental anomalies
        flux = np.nan_to_num(flux, nan=0.0)
        
        # 4. Z-score normalization across the temporal axis
        mean_flux = np.mean(flux, axis=0, keepdims=True)
        std_flux = np.std(flux, axis=0, keepdims=True) + 1e-8
        normalized_flux = (flux - mean_flux) / std_flux
        
        # 5. Reshape for PyTorch Conv3D: (Channels, Depth/Time, Height, Width)
        tensor_3d = torch.tensor(normalized_flux, dtype=torch.float32).unsqueeze(0)
        
        # Placeholder label (1.0 = Exoplanet, 0.0 = False Positive)
        label = torch.tensor(1.0, dtype=torch.float32)
        
        return tensor_3d, label

if __name__ == "__main__":
    dataset = TESSTensorDataset(data_dir="data")
    print(f"[*] Found {len(dataset)} TPFs in dataset.")
    
    if len(dataset) > 0:
        print(f"[+] Reading file: {dataset.file_list[0]}")
        x, y = dataset[0]
        print(f"[+] Output Tensor Shape: {list(x.shape)}")
        print(f"[+] Expected DataLoader Batch Shape: (Batch, {x.shape[0]}, {x.shape[1]}, {x.shape[2]}, {x.shape[3]})")