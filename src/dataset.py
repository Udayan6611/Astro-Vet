import os
import torch
import numpy as np
import pandas as pd
import lightkurve as lk
from torch.utils.data import Dataset

class TESSTensorDataset(Dataset):
    # Added target_shape constraint: (Time, Height, Width)
    def __init__(self, csv_file="data/sandbox_labels.csv", data_dir="data", target_shape=(18000, 15, 15)):
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.valid_data = []
        
        if not os.path.exists(csv_file):
            print(f"[-] {csv_file} not found. Run build_sandbox.py first.")
            return

        labels_df = pd.read_csv(csv_file)
        
        available_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".fits") or file.endswith(".fits.gz"):
                    available_files.append(os.path.join(root, file))
        
        print("[*] Cross-referencing CSV labels with downloaded files...")
        for _, row in labels_df.iterrows():
            tic_id = str(row['tic_id'])
            label = float(row['label'])
            
            matching_file = next((f for f in available_files if tic_id in f), None)
            
            if matching_file:
                try:
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
        
        tpf = lk.TessTargetPixelFile(file_path, quality_bitmask='hard')
        flux = tpf.flux.value
        flux = np.nan_to_num(flux, nan=0.0)
        
        mean_flux = np.mean(flux, axis=0, keepdims=True)
        std_flux = np.std(flux, axis=0, keepdims=True) + 1e-8
        normalized_flux = (flux - mean_flux) / std_flux
        
        # --- NEW: Pad or Crop to Fixed Shape ---
        target_t, target_h, target_w = self.target_shape
        t, h, w = normalized_flux.shape
        
        # 1. Temporal Axis (Time)
        if t > target_t:
            normalized_flux = normalized_flux[:target_t, :, :]
        elif t < target_t:
            normalized_flux = np.pad(normalized_flux, ((0, target_t - t), (0, 0), (0, 0)), mode='constant')
            
        # 2. Spatial Axis (Height)
        if h > target_h:
            normalized_flux = normalized_flux[:, :target_h, :]
        elif h < target_h:
            normalized_flux = np.pad(normalized_flux, ((0, 0), (0, target_h - h), (0, 0)), mode='constant')
            
        # 3. Spatial Axis (Width)
        if w > target_w:
            normalized_flux = normalized_flux[:, :, :target_w]
        elif w < target_w:
            normalized_flux = np.pad(normalized_flux, ((0, 0), (0, 0), (0, target_w - w)), mode='constant')
        # ---------------------------------------

        tensor_3d = torch.tensor(normalized_flux, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32) 
        
        return tensor_3d, label_tensor

if __name__ == "__main__":
    dataset = TESSTensorDataset()
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"[+] Output Tensor Shape: {list(x.shape)}")
        print(f"[+] Label Vector: {y}")