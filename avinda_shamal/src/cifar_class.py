import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    
    '''A custom dataset class for loading images from a directory structure where each class has its own subdirectory.'''
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        
        """
        Initializes the CustomImageDataset.

        Args:
            root_dir (str): Path to the root directory containing class subdirectories.
            transform (callable, optional): A function/transform to apply to the images (default: None).
            target_transform (callable, optional): A function/transform to apply to the labels (default: None).
            
        Returns: None
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append((img_path, class_idx)) # store image path and class folder

    def __len__(self):
        
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        
        return len(self.data)

    def __getitem__(self, idx):
        
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and its corresponding label.
        """
        
        img_path, label = self.data[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
