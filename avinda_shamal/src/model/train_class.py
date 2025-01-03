import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    """A class to train and test a PyTorch model."""

    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ):
        """Initializes the ModelTrainer.
        Args:
            network : The neural network model to train.
            train_loader : The DataLoader object for the training dataset.
            test_loader : The DataLoader object for the test dataset.
            device : The device to use for training and testing (cpu or cuda).
        """
        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train_model(
        self, epochs: int, criterian: nn.Module, optimizer: optim.Optimizer
    ) -> tuple:
        """Trains the model on the training dataset.
        Args:
            epochs : The number of epochs to train the model.
            criterian : The loss function to use.
            optimizer : The optimizer to use for training.
        Returns:
            tuple: The trained model and a dictionary of metrics.
        """
        self.network.to(self.device)
        metrics = {"epoch_loss": []}
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(
                self.train_loader, 0
            ):  # each iteration, trainloader yields a batch of data
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.network(inputs)  # forward pass
                loss = criterian(outputs, labels)  # calculate the loss
                loss.backward()  # backward pass
                optimizer.step()  # update the weights
                running_loss += loss.item()
            avg_loss = running_loss / len(self.train_loader)
            metrics["epoch_loss"].append(avg_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        print("Finished Training")
        return self.network, metrics

    def test_model(self) -> float:
        """Tests the model on the test dataset.
        Returns:
            float: The accuracy of the model on the test dataset.
        """
        self.network.to(self.device)
        correct, total = 0, 0
        with torch.no_grad():
            for test_data in self.test_loader:
                test_images, test_labels = test_data
                test_images, test_labels = (
                    test_images.to(self.device),
                    test_labels.to(self.device),
                )
                test_outputs = self.network(test_images)
                _, predicted = torch.max(
                    test_outputs, 1
                )  # choose the class which has highest energy is the predicted class
                # _ gives the energy value for predicted class which is not used here
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
            accuracy = correct / total * 100
        return accuracy
