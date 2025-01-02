import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model.nn_model import NNModel
import numpy as np


def train_model(network, device, epochs):
    network.to(device)
    metrics = {"epoch_loss": []}
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(
            train_loader, 0
        ):  # each iteration, trainloader yields a batch of data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = network(inputs)  # forward pass
            loss = criterian(outputs, labels)  # calculate the loss
            loss.backward()  # backward pass
            optimizer.step()  # update the weights
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        metrics["epoch_loss"].append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    print("Finished Training")
    return network, metrics


def test_model(network, device):
    network.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            test_images, test_labels = (
                test_images.to(device),
                test_labels.to(device),
            )
            test_outputs = network(test_images)
            _, predicted = torch.max(
                test_outputs, 1
            )  # choose the class which has highest energy is the predicted class
            # _ gives the energy value for predicted class
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
        accuracy = correct / total * 100
    return accuracy


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

# train the model
model, train_metrics = train_model(model, device, 15)

# test the model
score = test_model(model, device)
print(f"Accuracy of the network on the 10000 test images: {score:.2f} %")

# save the trained model
path = "./src/model/cifar10_net.pth"
torch.save(
    {
        "Trained_Model": model.state_dict(),
        "Epoch_Losses": train_metrics,
        "Test_Accuracy": score,
    },
    path,
)
print(f"Model and metrics saved to {path}")
