import torch
import torch.nn as nn

class AstroVet3DCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AstroVet3DCNN, self).__init__()
        
        # 1. Spatiotemporal Feature Extraction
        # Captures local transit patterns across time and pixel space
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(4, 1, 1))  # Aggressively downsample the time axis
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(4, 1, 1))
        )

        # 2. Adaptive Pooling
        # TESS pixel cutouts vary in size (e.g., 11x11, 21x11). 
        # This layer forces all outputs to a uniform (64, 1, 1) dimension.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((64, 1, 1))
        
        # 3. Bayesian Uncertainty Module (Monte Carlo Dropout)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 4. Binary Classifier
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 1 * 1, 128),
            nn.ReLU(),
            self.dropout,  # Kept active during inference for Monte Carlo sampling
            nn.Linear(128, 1),
            nn.Sigmoid()   # Output probability: 1.0 (Planet) to 0.0 (False Positive)
        )

    def forward(self, x):
        # Expected input shape: [Batch, Channels=1, Time, Height, Width]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Pool temporal and spatial dimensions to uniform size
        x = self.adaptive_pool(x)
        
        # Flatten the 3D tensor for the dense layers
        x = x.view(x.size(0), -1)
        
        # Generate classification probability
        out = self.fc(x)
        return out

if __name__ == "__main__":
    # Test a dummy tensor mirroring your dataset output [Batch, C, T, H, W]
    dummy_input = torch.randn(1, 1, 18264, 21, 11)
    
    model = AstroVet3DCNN(dropout_rate=0.5)
    output = model(dummy_input)
    
    print("[*] 3D-CNN Model Compiled Successfully.")
    print(f"[+] Output Probability: {output.item():.4f}")