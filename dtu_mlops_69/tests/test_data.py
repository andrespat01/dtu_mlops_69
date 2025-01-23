from tests import _PATH_DATA
from src.mlops_project.data import tweets
from torch.utils.data import Dataset
import torch
import os.path
import pytest

# Test the data module


@pytest.mark.skipif(not os.path.exists("data/processed/"), reason="Data files not found")
def test_data():
    train_set, test_set = tweets()
    assert len(
        train_set) == 9096, "Dataset did not have the correct number of training samples"
    assert len(
        test_set) == 2274, "Dataset did not have the correct number of test samples"

    # check the dataset is correct
    for dataset in [train_set, test_set]:
        for idx, (input_ids, attention_mask, targets) in enumerate(dataset):
            # Check the expected shapes and valid values
            assert input_ids.shape == (61,)  # should be 60 after padding
            assert attention_mask.shape == (61,)  # should be 60 after padding
            assert targets in [0, 1]  # Label should be either 0 or 1
