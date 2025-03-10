import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def rgb2lab(rgb_image): 

    image = rgb_image.permute(1,2,0)

    # Convert to Lab color space
    lab_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.float32)
        
    # Split channels
    l = lab_image[:, :, 0] / 100 #normalize in [0,1]
    ab = lab_image[:, :, 1:] / 128  # Normalize ab to [-1,1]

    return l,ab
    
 

def lab2rgb(l_channel,ab_channel):

    #denormalization 
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
    plt.xticks(np.arange(1,len(tlosses) + 1,5))
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