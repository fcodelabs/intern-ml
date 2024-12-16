from datasets import load_dataset
from torchvision import transforms
from cvas.huggingface_dataclass import HuggingFaceDataset

# load the cifar10 dataset
hf_dataset = load_dataset("cifar10")

# initialize the HuggingFaceDataset class
train_data = HuggingFaceDataset(hf_dataset, "train", transform=transforms.ToTensor())
test_data = HuggingFaceDataset(hf_dataset, "test", transform=transforms.ToTensor())

print(len(test_data))
print(len(train_data))
print(train_data[0])
print(train_data[0][0].shape)
