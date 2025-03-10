import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import config
from tqdm import tqdm
from random import randint
import torchvision.transforms as tf

def rgb2lab(rgb_image): 

    image = rgb_image.permute(1,2,0)

    # Convert to Lab color space
    lab_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.float32)
        
    # Split channels
    l = lab_image[:, :, 0] / 100 #normalize in [0,1]
    ab = lab_image[:, :, 1:] / 128  # Normalize ab to [-1,1]

    return l,ab
    
def lab2rgb(l_channel,ab_channel):

    # Step 1: denormalization 
    ab_channel *= 128

    # Step 2: Merge L and AB into (H, W, 3) format
    Lab_image = torch.cat([l_channel, ab_channel], dim=0).cpu().permute(1, 2, 0).numpy().astype(np.float32)  # (H, W, 3)

    # Step 3: Convert Lab → RGB using OpenCV
    RGB_image = cv2.cvtColor(Lab_image, cv2.COLOR_Lab2RGB)

    return RGB_image

def plot_loss(tlosses,vlosses, title="Loss Curve", xlabel="Epoch", ylabel="Loss"):
    plt.style.use('bmh')
    # Create the plot
    plt.figure(figsize=(8, 5))

    epochs=np.arange(1,len(tlosses) + 1)
    
    plt.plot(epochs, tlosses, marker='o', linestyle='-', color='royalblue', label='Train Loss')
    plt.plot(epochs, vlosses, marker='s', linestyle='--', color='darkorange', label="Validation Loss")
    
    # Labels & title
    ticks=np.arange(5,len(tlosses) + 1,5)
    plt.xticks(np.insert(ticks,0,1))
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.savefig(f"figures/loss.png")

    # Show the plot
    plt.show()

def plot_images(grayscale_images, colorized_images, ground_truth_images, num_rows=5):
    """
    Plots a comparison of grayscale, colorized, and ground truth images in a grid.

    Args:
        grayscale_images: List of grayscale images.
        colorized_images: List of colorized images.
        ground_truth_images: List of ground truth images.
        num_rows: Number of rows to display (default is 5).
    """
    # Set up figure layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(9, num_rows * 2), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

    # Set column titles (only at the top)
    column_titles = ["Grayscale", "Colorized", "Ground Truth"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    # Plot images
    for i in range(num_rows):
        axes[i, 0].imshow(grayscale_images[i], cmap="gray")
        axes[i, 1].imshow(colorized_images[i])  # Assuming RGB format
        axes[i, 2].imshow(ground_truth_images[i])

        # Remove axes for a cleaner layout
        for j in range(3):
            axes[i, j].axis("off")

    plt.show()

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="checkpoint.pth", verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Path to save the best model checkpoint.
            verbose (bool): Whether to print improvement messages.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = 1.0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter if validation loss improves

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

def train_model(model,epochs,device,train_loader,val_loader,criterion,optimizer,scheduler):

    tloss_list=[] #for plotting
    vloss_list=[]

    # === Training Loop === #
    for epoch in range(epochs):

        model.train()  # Set model to training mode
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

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

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

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
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        tloss_list.append(train_loss)
        vloss_list.append(val_loss)

        # Call early stopping
        scheduler(val_loss, model)

        if scheduler.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
    
    print("✅ Training Complete!")
    plot_loss(tloss_list,vloss_list)

def test_model(model, device, test_loader, input_mode):

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

        idx=randint(0,min(config.BATCH_SIZE,len(test_loader.dataset)-1))
        output_ab = AB_pred[idx]

        input_l = L[idx] * 100 #normalize in [0,1]
        input_ab_sample = AB[idx]
             
        colorized = lab2rgb(input_l,output_ab)
        ground_truth = lab2rgb(input_l,input_ab_sample)

        input_imgs.append(input_l.cpu().squeeze(0))
        output_imgs.append(colorized)
        gt_imgs.append(ground_truth)

    plot_images(input_imgs,output_imgs,gt_imgs)
