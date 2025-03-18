import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


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

def plot_loss(tlosses,vlosses, title="Loss Curve", xlabel="Epochs", ylabel="Loss", fname="plots/loss.png"):
    plt.style.use('bmh')
    # Create the plot
    plt.figure(figsize=(8, 5))

    epochs=np.arange(1,len(tlosses) + 1)
    
    plt.plot(epochs,tlosses, marker='o', linestyle='-', color='blue', label='Train Loss')
    plt.plot(epochs,vlosses, marker='s', linestyle='--', color='red', label="Validation Loss")
    
    # Labels & title
    ticks=np.arange(5,len(tlosses) + 1,5)
    plt.xticks(np.insert(ticks,0,1))
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.savefig(fname)

    # Show the plot
    plt.show()

def plot_images(grayscale_images, colorized_images, ground_truth_images):
    """
    Plots a comparison of grayscale, colorized, and ground truth images in a grid.

    Args:
        grayscale_images: List of grayscale images.
        colorized_images: List of colorized images.
        ground_truth_images: List of ground truth images.
        num_rows: Number of rows to display (default is 5).
    """

    num_rows=len(grayscale_images)
    
    # Set up figure layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(9, num_rows * 2), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

    # Set column titles (only at the top)
    column_titles = ["Grayscale", "Colorized(GAE)", "Ground Truth"]

    if num_rows == 1:
        for col, title in enumerate(column_titles):
            axes[col].set_title(title, fontsize=12, fontweight="bold")

        axes[0].imshow(grayscale_images[0], cmap="gray")
        axes[1].imshow(colorized_images[0])  # Assuming RGB format
        axes[2].imshow(ground_truth_images[0])

        # Remove axes for a cleaner layout
        for j in range(3):
            axes[j].axis("off")
    
    else:
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

def plot_prediction(input,prediction):

    # Set up figure layout
    fig, axes = plt.subplots(1, 2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

    # Set column titles (only at the top)
    column_titles = ["Grayscale", "Colorized"]

    for col, title in enumerate(column_titles):
        axes[col].set_title(title, fontsize=12, fontweight="bold")

    axes[0].imshow(input, cmap="gray")
    axes[1].imshow(prediction)  # Assuming RGB format
    
    # Remove axes for a cleaner layout
    for j in range(2):
        axes[j].axis("off")

    plt.show()
    

