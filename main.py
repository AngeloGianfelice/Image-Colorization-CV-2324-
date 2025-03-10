import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from colorizers import ColorizationAutoencoder,VGGAutoencoder
from datasets import Cocostuff_Dataset
from utils import EarlyStopping,train_model,test_model

# Create datasets
train_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="train")
val_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="val")
test_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="test")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print(f"✅ Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ColorizationAutoencoder().to(device)

# Freeze encoder weights
#for param in model.encoder.parameters():
    #param.requires_grad = False  # Encoder is fixed (pretrained weights)

# === Define Training Components === #
criterion = nn.MSELoss()  # Loss Function
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # Optimizer
early_stopping = EarlyStopping(patience=config.PATIENCE, path=f"models/best_model_{config.EPOCHS}.pth", verbose=True) # Early Stopping

# === Train Colorization Model === #
train_model(model=model, epochs=config.EPOCHS, device=device, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=early_stopping)

# Load Best Model for Testing
model.load_state_dict(torch.load(f"models/best_model_{config.EPOCHS}.pth"))
test_model(model=model, device=device, test_loader=test_loader)