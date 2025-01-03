import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ModelTrainer:
    """A class to train and test a PyTorch model."""

    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
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
        self,
        epochs: int,
        criterian: nn.Module,
        optimizer: optim.Optimizer,
        split: float = 0.8,
    ) -> tuple:
        """Trains the model on the training dataset.
        Args:
            epochs : The number of epochs to train the model.
            criterian : The loss function to use.
            optimizer : The optimizer to use for training.
            split : The percentage of the training data to use for validation.
        Returns:
            tuple: The trained model and a dictionary of metrics.
        """
        self.network.to(self.device)
        metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
        }
        train_size = int(len(self.train_loader.dataset) * split)
        val_size = len(self.train_loader.dataset) - train_size
        train_data, val_data = random_split(
            self.train_loader.dataset, [train_size, val_size]
        )
        train_set = DataLoader(
            train_data, batch_size=self.train_loader.batch_size, shuffle=True
        )
        val_set = DataLoader(
            val_data, batch_size=self.train_loader.batch_size, shuffle=False
        )
        for epoch in range(epochs):  # loop over the dataset multiple times
            for phase in ["train", "eval"]:
                if phase == "train":
                    self.network.train()
                    loader = train_set
                else:
                    self.network.eval()
                    loader = val_set
                running_loss = 0.0
                correct, total = 0, 0

                for (
                    data
                ) in loader:  # each iteration, loader yields a batch of data
                    inputs, labels = data
                    inputs, labels = (
                        inputs.to(self.device),
                        labels.to(self.device),
                    )
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.network(inputs)  # forward pass
                        loss = criterian(outputs, labels)  # calculate the loss
                        if phase == "train":
                            loss.backward()  # backward pass
                            optimizer.step()  # update the weights
                    running_loss += loss.item() * inputs.size(
                        0
                    )  # Accumulate loss
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (
                        (predicted == labels).sum().item()
                    )  # count correct predictions
                epoch_loss = running_loss / len(loader.dataset)
                epoch_accuracy = correct / total * 100
                if phase == "train":
                    metrics["train_loss"].append(epoch_loss)
                    metrics["train_accuracy"].append(epoch_accuracy)
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], "
                        f"Train Loss: {metrics['train_loss'][-1]:.4f}, Train Accuracy: {metrics['train_accuracy'][-1]:.2f}%"
                    )
                else:
                    metrics["eval_loss"].append(epoch_loss)
                    metrics["eval_accuracy"].append(epoch_accuracy)
                    print(
                        f"Val Loss: {metrics['eval_loss'][-1]:.4f}, Val Accuracy: {metrics['eval_accuracy'][-1]:.2f}%"
                    )
                # Check early stopping
                should_stop, best_loss = self.early_stopping(
                    epoch_loss, best_loss
                )
                if should_stop:
                    return metrics
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

    def early_stopping(
        self,
        val_loss: float,
        best_loss: float = float("inf"),
        patience_counter: int = 0,
        patience: int = 2,
    ) -> tuple:
        """Checks if early stopping should be triggered.
        Args:
            val_loss (float): Current epoch's validation loss.
            best_loss (float): Best validation loss observed so far.
            patience_counter (int): Current count of epochs without improvement.
            patience (int): Maximum allowed epochs without improvement.
        Returns:
            tuple: where should stop and updated_best_loss
        """
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                return True, best_loss
        return False, best_loss

    def learning_curves(self, metrics: dict) -> None:
        """Plots the learning curves for the model.
        Args:
            metrics : A dictionary containing the training loss and accuracy.
        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(metrics["train_loss"], label="Training Loss")
        plt.plot(metrics["eval_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(metrics["train_accuracy"], label="Training Accuracy")
        plt.plot(metrics["eval_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()
        plt.show()
