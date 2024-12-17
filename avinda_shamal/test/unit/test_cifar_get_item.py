from unittest.mock import patch
from cvas.dataclasses import CustomImageDataset
import torch
from torchvision import transforms

transform = transforms.Resize((32, 32))


def test_get_item_custom_dataset():
    # Initialize the dataset and set up mock data
    with patch("cvas.dataclasses.read_image") as mock_read_image:
        mock_read_image.return_value = torch.rand(3, 44, 44)

        with patch("cvas.dataclasses.CustomImageDataset.__init__", return_value=None):
            dataset = CustomImageDataset(
                root_dir="mock_path", transform=transform, target_transform=None
            )
            dataset.transform = transform
            dataset.target_transform = None
            dataset.data = [
                ("a.jpg", "horse"),
                ("b.jpg", "frog"),
                ("c.jpg", "dog"),
                ("d.jpg", "cat"),
            ]

            # test the get_item method
            img, label = dataset[0]
            assert label == "horse"
            assert isinstance(img, torch.Tensor)
            assert img.shape == (3, 32, 32)
