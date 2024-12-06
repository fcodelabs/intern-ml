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
    
# Define transformations (resize to 32x32 and normalize)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Apply standart Normalization and then convert to [-1, 1] range
    # transforms.ToTensor()  # convert each image to a tensor and normalize to [0, 1] range
    # Means and Standard deviations of R,G,B planes are 0.5
    
    ### no need to apply ToTensor() as read_image() already converts image to tensor
])

# Load the training and test data
training_data = CustomImageDataset("D:/Intern ML/cifar10/train", transform=transform)
test_data = CustomImageDataset("D:/Intern ML/cifar10/test", transform=transform)

print(len(training_data))
   
# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

print(len(train_dataloader))

labels_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    img_ = img.permute(1, 2, 0).numpy()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img_)
plt.show()

## Another method to display image and label
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()
print(f"Label: {labels_map[label.item()]}")