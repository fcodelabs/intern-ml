from cvas.huggingface_dataclass import HuggingFaceDataset
import torch
from torchvision import transforms

transform = transforms.Resize((32, 32))


def test_hf_get_item():
    hf_dataset = {
        "train": [
            {"img": torch.rand(3, 44, 44), "label": "dog"},
            {"img": torch.rand(3, 96, 96), "label": "cat"},
            {"img": torch.rand(3, 100, 100), "label": "ship"},
            {"img": torch.rand(3, 80, 80), "label": "truck"},
        ],
        "test": [
            {"img": torch.rand(3, 36, 36), "label": "airplane"},
            {"img": torch.rand(3, 58, 58), "label": "deer"},
        ],
    }
    dataset1 = HuggingFaceDataset(hf_dataset, "train", transform=transform)
    assert dataset1[1][0].shape == (3, 32, 32)
    assert dataset1[2][1] == "ship"
    assert isinstance(dataset1[3][0], torch.Tensor)

    dataset2 = HuggingFaceDataset(hf_dataset, "test", transform=transform)
    assert dataset2[0][1] == "airplane"
    assert dataset2[1][0].shape == (3, 32, 32)
