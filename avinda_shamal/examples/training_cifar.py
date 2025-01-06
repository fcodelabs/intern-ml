import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import NNModel
from model import ModelTrainer
from model import ModelEvaluator

# load and normalize the Cifar10 dataset
std = np.array([0.5, 0.5, 0.5])
mean = np.array([0.5, 0.5, 0.5])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)
train_set = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)

test_set = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=0)
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
# define the model, device, loss function, and optimizer
model = NNModel(3, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = ModelTrainer(model, train_loader, device)

# train the model
model, train_metrics = trainer.train_model(
    epochs=15, criterian=criterian, optimizer=optimizer
)
# plot the learning curves
trainer.learning_curves(train_metrics)

# test the model
tester = ModelEvaluator(model, test_loader)
test_metrics = tester.test_model()

# save the trained model
path = "./src/model/cifar10_net.pth"
torch.save(
    {
        "Trained_Model": model.state_dict(),
        "Epoch_Losses": train_metrics,
        "Test_Accuracy": test_metrics,
    },
    path,
)
print(f"Model and metrics saved to {path}")
