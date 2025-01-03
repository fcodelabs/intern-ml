import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(
        self,
        network: nn.Module,
        test_loader: DataLoader,
    ):
        """test_loader : The DataLoader object for the test dataset."""
        self.network = network
        self.test_loader = test_loader
        self.y_true = []
        self.y_pred = []
        self.y_probs = []

    def test_model(self) -> dict:
        """
        Evaluates the model on the test dataset and calculates various metrics.

        Returns:
            dict: A dictionary containing evaluation metrics such as accuracy, precision, recall, F1 score,
                  confusion matrix, and optional ROC-AUC score.
        """
        metrics = {}
        self.network.eval()
        with torch.no_grad():
            for test_data in self.test_loader:
                test_images, test_labels = test_data
                test_outputs = self.network(test_images)
                _, predicted = torch.max(test_outputs, 1)
                # choose the class which has highest energy is the predicted class
                # _ gives the energy value for predicted class which is not used here
                self.y_true.extend(test_labels)
                self.y_pred.extend(predicted.numpy())
                # Calculate probabilities (for AUC)
                probabilities = F.softmax(
                    test_outputs, dim=1
                )  # Multiclass softmax
                self.y_probs.extend(
                    probabilities.numpy()
                )  # Append probabilities

        metrics["Accuracy"] = accuracy_score(self.y_true, self.y_pred)
        metrics["precision"] = precision_score(
            self.y_true, self.y_pred, average="weighted"
        )
        metrics["recall"] = recall_score(
            self.y_true, self.y_pred, average="weighted"
        )
        metrics["f1_score"] = f1_score(
            self.y_true, self.y_pred, average="weighted"
        )
        metrics["confusion_matrix"] = confusion_matrix(
            self.y_true, self.y_pred
        )
        # AUC
        if self.y_probs is not None:
            metrics["roc_auc"] = roc_auc_score(
                self.y_true, self.y_probs, multi_class="ovr"
            )
        # Print the metrics
        for key, value in metrics.items():
            if key == "confusion_matrix":
                print(f"{key}:\n{value}")
            else:
                print(f"{key}: {value:.4f}")

        # Plot ROC Curve
        if self.y_probs is not None:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_probs)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label="ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate (Recall)")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend()
            plt.show()

        return metrics
