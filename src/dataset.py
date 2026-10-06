import os
import torch
import numpy as np
import pandas as pd
import lightkurve as lk
from torch.utils.data import Dataset

class TESSTensorDataset(Dataset):
    def __init__(self, csv_file="data/sandbox_labels.csv", data_dir="data"):
        self.data_dir = data_dir
        self.valid_data = []
        
        if not os.path.exists(csv_file):
            print(f"[-] {csv_file} not found. Run build_sandbox.py first.")
            return

        labels_df = pd.read_csv(csv_file)
        
        # Map all available downloaded FITS files
        available_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".fits") or file.endswith(".fits.gz"):
                    available_files.append(os.path.join(root, file))
        
        print("[*] Cross-referencing CSV labels with downloaded files...")
        for _, row in labels_df.iterrows():
            tic_id = str(row['tic_id'])
            label = float(row['label'])
            
            # Find the downloaded file that belongs to this TIC
            matching_file = next((f for f in available_files if tic_id in f), None)
            
            if matching_file:
                try:
                    # Test if the file is readable (catches corrupted/truncated downloads)
                    tpf = lk.TessTargetPixelFile(matching_file)
                    self.valid_data.append((matching_file, label))
                except Exception:
                    print(f"[-] File for TIC {tic_id} is corrupted. Dropping from dataset.")
            else:
                print(f"[-] Missing file for TIC {tic_id}. Dropping from dataset.")
                
        print(f"[+] Successfully verified {len(self.valid_data)} clean samples for training.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        file_path, label = self.valid_data[idx]
        
        # 1. Load the TPF
        tpf = lk.TessTargetPixelFile(file_path, quality_bitmask='hard')
        
        # 2. Extract and clean the 3D flux array
        flux = tpf.flux.value
        flux = np.nan_to_num(flux, nan=0.0)
        
        # 3. Z-score normalization across the temporal axis
        mean_flux = np.mean(flux, axis=0, keepdims=True)
        std_flux = np.std(flux, axis=0, keepdims=True) + 1e-8
        normalized_flux = (flux - mean_flux) / std_flux
        
        # 4. Reshape for PyTorch Conv3D: (Channels, Depth/Time, Height, Width)
        tensor_3d = torch.tensor(normalized_flux, dtype=torch.float32).unsqueeze(0)
        
        # Output as a shape [1] tensor for BCE Loss
        label_tensor = torch.tensor([label], dtype=torch.float32) 
        
        return tensor_3d, label_tensor

if __name__ == "__main__":
    # Test the mapping and label extraction
    dataset = TESSTensorDataset()
    
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"[+] Output Tensor Shape: {list(x.shape)}")
        print(f"[+] Label Vector: {y}")