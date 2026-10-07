import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import TESSTensorDataset
from model import AstroVet3DCNN

def train_model():
    # 1. Setup Device (Automatically use Nvidia GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 2. Load Dataset
    print("[*] Loading Sandbox Dataset...")
    dataset = TESSTensorDataset(csv_file="data/sandbox_labels.csv", data_dir="data")
    
    if len(dataset) == 0:
        print("[-] No data found. Aborting training.")
        return
        
    # Batch size is small (2) because 3D telescope data is memory-heavy
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = AstroVet3DCNN(dropout_rate=0.5).to(device)
    criterion = nn.BCELoss()  # Binary classification (Planet vs False Positive)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 5
    print(f"\n[*] Starting Training Loop for {epochs} Epochs...")
    
    for epoch in range(epochs):
        model.train()  # Enables Monte Carlo Dropout
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass (calculate gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"[+] Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    # 5. Save the trained weights
    torch.save(model.state_dict(), "astrovet_prototype.pth")
    print("\n[+] Training Complete! Model weights saved to 'astrovet_prototype.pth'")

if __name__ == "__main__":
    train_model()