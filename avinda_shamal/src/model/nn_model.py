import torch
import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, in_fetures: int, out_features: int):
        """Initializes the neural network model

        Args: in_fetures, out_features
        Returns: None
        """
        super(NNModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_fetures, out_channels=8, kernel_size=3, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1
        )

        self.fc1 = nn.Linear(in_features=6 * 6 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args: input tensor
        Returns: output tensor
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 6 * 6 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x
