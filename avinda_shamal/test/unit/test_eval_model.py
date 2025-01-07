import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import ModelEvaluator
import numpy as np


def test_test_model():
    # Create a small synthetic dataset
    test_data = torch.tensor(
        [
            [0.1, 0.71, 0.18, 0.01],
            [0.65, 0.20, 0.09, 0.06],
            [0.27, 0.15, 0.03, 0.55],
            [0.1, 0.71, 0.18, 0.01],
            [0.65, 0.20, 0.09, 0.06],
            [0.27, 0.15, 0.03, 0.55],
            [0.1, 0.71, 0.18, 0.01],
            [0.65, 0.20, 0.09, 0.06],
            [0.27, 0.15, 0.03, 0.55],
            [0.1, 0.71, 0.18, 0.01],
        ]
    )
    test_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dummy_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
    evaluator = ModelEvaluator(nn.Linear(4, 10), test_loader)
    # Test the method
    metrics = evaluator.test_model()
    # Validate metrics
    assert metrics["Accuracy"] >= 0
    assert isinstance(metrics["confusion_matrix"], np.ndarray)
    assert "precision" in metrics and metrics["precision"] > 0
    assert metrics["recall"] > 0
    assert metrics["f1_score"] > 0
