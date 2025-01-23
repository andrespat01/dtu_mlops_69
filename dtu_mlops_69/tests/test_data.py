from tests import _PATH_DATA
from src.mlops_project.data import tweets, MyDataset, preprocess
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

# Test data preprocessing (check if the correct files are saved)
def test_preprocess():
    raw_data_path = "data/raw/tweets.csv"
    output_folder = "data/processed"

    # Ensure the raw data exists
    assert os.path.exists(raw_data_path), f"Raw data file {raw_data_path} not found"
    
    # Preprocess the data
    preprocess(raw_data_path, output_folder)
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)
    
    # Check if the processed files are saved
    assert os.path.exists(os.path.join(output_folder, "input_ids.pt")), "input_ids file not saved"
    assert os.path.exists(os.path.join(output_folder, "attention_mask.pt")), "attention_mask file not saved"
    assert os.path.exists(os.path.join(output_folder, "targets.pt")), "targets file not saved"