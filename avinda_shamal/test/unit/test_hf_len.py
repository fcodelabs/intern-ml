from cvas.huggingface_dataclass import HuggingFaceDataset
import pytest
from datasets import Dataset, DatasetDict


@pytest.mark.parametrize(
    "hf_dataset, split, expected",
    [
        (
            DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {
                            "img": ["image1", "image2", "image3", "image4"],
                            "label": ["label1", "label2", "label3", "label4"],
                        },
                    ),
                    "test": Dataset.from_dict(
                        {
                            "img": ["image5"],
                            "label": ["label5"],
                        },
                    ),
                }
            ),
            "train",
            4,
        ),
        (
            DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {
                            "img": ["image1", "image2", "image3"],
                            "label": ["label1", "label2", "label3"],
                        },
                    ),
                    "test": Dataset.from_dict(
                        {
                            "img": ["image4", "image5"],
                            "label": ["label4", "label5"],
                        },
                    ),
                }
            ),
            "test",
            2,
        ),
    ],
)
def test_hf_len(hf_dataset, split, expected):
    dataset = HuggingFaceDataset(hf_dataset, split, transform=None)
    assert len(dataset) == expected
