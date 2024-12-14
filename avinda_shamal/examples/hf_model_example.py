from datasets import load_dataset
from torchvision import transforms
from package_avinda_shamal.huggingface_dataclass import CustomImageDataset

# load the cifar10 dataset
hf_dataset = load_dataset("cifar10")

# initialize the CustomImageDataset class
train_data = CustomImageDataset(hf_dataset, "train", transform=transforms.ToTensor())
test_data = CustomImageDataset(hf_dataset, "test", transform=transforms.ToTensor())

print(len(test_data))
print(len(train_data))
print(train_data[0])
print(train_data[0][0].shape)
