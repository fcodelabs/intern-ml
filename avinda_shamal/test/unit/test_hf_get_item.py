from cvas.huggingface_dataclass import HuggingFaceDataset
import torch
from torchvision import transforms
from datasets import Dataset, DatasetDict
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])


def test_hf_get_item():
    hf_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "img": [
                        Image.new("RGB", (44, 44), (100, 50, 90)),
                        Image.new("RGB", (96, 96), (120, 32, 10)),
                        Image.new("RGB", (200, 200), (109, 50, 200)),
                        Image.new("RGB", (80, 80), (10, 150, 60)),
                    ],
                    "label": ["car", "ship", "truck", "airplane"],
                },
            ),
            "test": Dataset.from_dict(
                {
                    "img": [
                        Image.new("RGB", (150, 150), (30, 170, 220)),
                        Image.new("RGB", (58, 58), (0, 50, 255)),
                    ],
                    "label": ["airplane", "deer"],
                },
            ),
        }
    )

    dataset1 = HuggingFaceDataset(hf_dataset, "train", transform=transform)
    assert dataset1[0][0].shape == (3, 32, 32)
    assert dataset1[1][1] == "ship"
    assert isinstance(dataset1[3][0], torch.Tensor)

    dataset2 = HuggingFaceDataset(hf_dataset, "test", transform=transform)
    assert dataset2[0][1] == "airplane"
    assert dataset2[1][0].shape == (3, 32, 32)
