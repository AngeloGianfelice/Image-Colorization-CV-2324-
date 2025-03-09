import matplotlib.pyplot as plt
import numpy as np
import config

def rgb2lab(): #TODO
    pass 

def lab2rgb(): #TODO
    pass 

def plot_loss(tlosses,vlosses, title="Loss Curve", xlabel="Epoch", ylabel="Loss"):
    plt.style.use('bmh')
    # Create the plot
    plt.figure(figsize=(8, 5))

    epochs=np.arange(1,len(tlosses) + 1)
    
    plt.plot(epochs, tlosses, marker='o', linestyle='-', color='royalblue', label='Train Loss')
    plt.plot(epochs, vlosses, marker='s', linestyle='--', color='darkorange', label="Validation Loss")
    
    # Labels & title
    plt.xticks(epochs)
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