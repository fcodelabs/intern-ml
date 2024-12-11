from unittest.mock import patch
from src.cifar_class import CustomImageDataset
import torch
from torchvision import transforms

transform = transforms.Compose([transforms.Resize(32)])


def test_get_item_custom_dataset():
    # Initialize the dataset and set up mock data
    with patch("src.cifar_class.read_image") as mock_read_image:
        mock_read_image.return_value = torch.rand(44, 44, 3)

        with patch("src.cifar_class.CustomImageDataset.__init__", return_value=None):
            dataset = CustomImageDataset("mock_path", transform=transform)
            dataset.data = [
                ("a.jpg", "horse"),
                ("b.jpg", "frog"),
                ("c.jpg", "dog"),
                ("d.jpg", "cat"),
            ]

            # test the get_item method
            img, label = dataset[0]
            assert label == "horse"
            assert img == torch.tensor
            assert img.shape == (32, 32, 3)
