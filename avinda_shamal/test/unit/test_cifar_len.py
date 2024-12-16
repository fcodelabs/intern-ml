import pytest
from unittest.mock import patch
from cvas.dataclasses import CustomImageDataset


@pytest.mark.parametrize(
    "datalist, expected",
    [
        (
            [("a.jpg", "horse"), ("b.jpg", "frog"), ("c.jpg", "dog"), ("d.jpg", "cat")],
            4,
        ),
        ([("a.jpg", "horse"), ("b.jpg", "frog"), ("c.jpg", "dog")], 3),
        ([("a.jpg", "horse"), ("b.jpg", "frog")], 2),
        ([("a.jpg", "horse")], 1),
    ],
)
def test_len_custom_dataset(datalist, expected):
    with patch("cvas.dataclasses.CustomImageDataset.__init__", return_value=None):
        dataset = CustomImageDataset(root_dir="mock_path", transform=None)
        dataset.data = datalist

        # test the len method
        assert len(dataset) == expected
