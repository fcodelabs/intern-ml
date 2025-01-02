import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model.nn_model import NNModel
import numpy as np

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
# define the model, loss function, and optimizer
model = NNModel(3, 10)
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(
        train_loader, 0
    ):  # each iteration, trainloader yields a batch of data
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)  # forward pass
        loss = criterian(outputs, labels)  # calculate the loss
        loss.backward()  # backward pass
        optimizer.step()  # update the weights
        running_loss += loss.item()
        if i % 3000 == 2999:  # print every 3000 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
            running_loss = 0
print("Finished Training")

# save the trained model
path = "./src/model/cifar10_net.pth"
torch.save(model.state_dict(), path)

# test the model
correct, total = 0, 0
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data
        test_outputs = model(test_images)
        _, predicted = torch.max(
            test_outputs, 1
        )  # choose the class which has highest energy is the predicted class
        # _ gives the energy value for predicted class
        total += labels.size(0)
        correct += (predicted == test_labels).sum().item()
print(
    f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
)
