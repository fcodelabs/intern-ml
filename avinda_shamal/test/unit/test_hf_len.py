from cvas.huggingface_dataclass import HuggingFaceDataset
import pytest


@pytest.mark.parametrize(
    "hf_dataset, split, expected",
    [
        (
            {
                "train": [
                    {"img": "image1", "label": "label1"},
                    {"img": "image2", "label": "label2"},
                    {"img": "image3", "label": "label3"},
                    {"img": "image4", "label": "label4"},
                ],
                "test": [
                    {"img": "image5", "label": "label5"},
                ],
            },
            "train",
            4,
        ),
        (
            {
                "train": [
                    {"img": "image1", "label": "label1"},
                    {"img": "image2", "label": "label2"},
                    {"img": "image3", "label": "label3"},
                ],
                "test": [
                    {"img": "image4", "label": "label4"},
                    {"img": "image5", "label": "label5"},
                ],
            },
            "test",
            2,
        ),
    ],
)
def test_hf_len(hf_dataset, split, expected):
    dataset = HuggingFaceDataset(hf_dataset, split, transform=None)
    assert len(dataset) == expected
