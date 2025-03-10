import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import config
from utils import rgb2lab
import cv2

class Cocostuff_Dataset(Dataset):
    def __init__(self, image_dir, phase="train", split_ratios=config.SPLIT_RATIO, image_size=config.IMG_SIZE, seed=config.SEED,input_mode='gray'):
        
        """
        Args:
            image_dir (str): Path to the directory containing all images.
            phase (str): One of ["train", "val", "test"].
            split_ratios (tuple): Ratios for train, val, and test (default: 80%, 10%, 10%).
            image_size (int): Size to which images are resized.
            seed (int): Random seed for reproducibility.
        """
        self.image_dir = image_dir
        self.phase = phase
        self.image_size = image_size
        self.input_mode = input_mode
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png'))]
        
        # Ensure reproducibility
        random.seed(seed)
        random.shuffle(self.image_paths)
        
        # Compute dataset splits
        total = len(self.image_paths)
        train_size = int(split_ratios[0] * total)
        val_size = int(split_ratios[1] * total)
        
        # Assign dataset splits
        if phase == "train":
            self.image_paths = self.image_paths[:train_size]
        elif phase == "val":
            self.image_paths = self.image_paths[train_size:train_size + val_size]
        elif phase == "test":
            self.image_paths = self.image_paths[train_size + val_size:]

        # Define transformations
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        if phase == "train":
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0))
            ])
        else:
            self.augment = None  # No augmentation for val/test

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path= self.image_paths[idx]  # Get image path
        image = Image.open(image_path).convert("RGB") 

        if self.phase == "train" and self.augment:
            image = self.augment(image)
        
        image = self.base_transform(image)
        
        L_channel,AB_channel=rgb2lab(image)

        if self.input_mode == 'rgb':
            l_rgb = cv2.cvtColor(L_channel, cv2.COLOR_GRAY2RGB)  # Shape: (224, 224, 3)
            L_tensor = torch.tensor(l_rgb).permute(2, 0, 1)

        elif self.input_mode == 'gray':
            L_tensor = torch.tensor(L_channel).unsqueeze(0)

        else: 
            print("Wrong input mode, Exiting...")
            exit()

        AB_tensor = torch.tensor(AB_channel).permute(2, 0, 1)

        return L_tensor, AB_tensor