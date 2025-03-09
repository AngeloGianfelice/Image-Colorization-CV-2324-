import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import config
from colorizers import ColorizationAutoencoder
from datasets import Cocostuff_Dataset
from utils import rgb2lab,lab2rgb,plot_loss,plot_images
from random import randint

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

# === 3. Define Training Components === #
criterion = nn.MSELoss()  # Loss Function
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # Optimizer

# === 4. Training Loop === #
best_val_loss = 1  # Track best validation loss
tloss_list=[]
vloss_list=[]

for epoch in range(config.EPOCHS):
    model.train()  # Set model to training mode
    train_loss = 0
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]", leave=False)
    for batch_idx,(image_l, image_ab) in enumerate(train_progress):
        image_l, image_ab = image_l.to(device), image_ab.to(device)
        optimizer.zero_grad()
        output = model(image_l)
        loss = criterion(output, image_ab)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # Update tqdm description with batch loss
        train_progress.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0

    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Validation]", leave=False)

    with torch.no_grad():
        for idx, (L, AB) in enumerate(val_progress):
            L, AB = L.to(device), AB.to(device)
            output = model(L)
            loss = criterion(output, AB)
            val_loss += loss.item()
            # Update tqdm description with batch loss
            val_progress.set_postfix(loss=loss.item())
    
    val_loss /= len(val_loader)

    # Print epoch summary
    print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"models/best_model_{config.EPOCHS}.pth")
        print("✅ Model Saved!")

    tloss_list.append(train_loss)
    vloss_list.append(val_loss)
    
print("✅ Training Complete!")
plot_loss(tloss_list,vloss_list)

# === 5. Testing Phase === #
def test_model(model, test_loader):
    model.eval()
    input_imgs=[]
    output_imgs=[]
    gt_imgs=[]
    # Run inference on test images
    dataiter=iter(test_loader)
    L, AB = next(dataiter)
    L = L.to(device)
    AB = AB.to(device)
        
    with torch.no_grad():
        AB_pred = model(L)

    for _ in range(config.NUM_TEST):
        idx=randint(0,config.BATCH_SIZE)
        input_l_sample = L[idx] * 100 # Scale L from [0,1] to [0,100]
        input_ab_sample = AB[idx] * 128   # Scale L from [0,1] to [0,100]
        output_ab_sample = AB_pred[idx] * 128  # Scale AB from [-1,1] to [-128,127]

        # Step 2: Merge L and AB into (H, W, 3) format
        Lab_image = torch.cat([input_l_sample, output_ab_sample], dim=0).cpu().permute(1, 2, 0).numpy().astype(np.float32)  # (H, W, 3)

        # Step 3: Convert Lab → RGB using OpenCV
        RGB_image = cv2.cvtColor(Lab_image, cv2.COLOR_Lab2RGB)

        # Step 2: Merge L and AB into (H, W, 3) format
        gt_lab = torch.cat([input_l_sample, input_ab_sample], dim=0).permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, 3)

        # Step 3: Convert Lab → RGB using OpenCV
        gt_rgb = cv2.cvtColor(gt_lab, cv2.COLOR_Lab2RGB)

        input_imgs.append(input_l_sample.cpu().squeeze(0))
        output_imgs.append(RGB_image)
        gt_imgs.append(gt_rgb)

    plot_images(input_imgs,output_imgs,gt_imgs)

# Load Best Model for Testing
model.load_state_dict(torch.load(f"models/best_model_{config.EPOCHS}.pth"))
test_model(model, test_loader)