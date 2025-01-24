from pathlib import Path
from google.cloud import storage
import pandas as pd
import re
import torch
import typer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import subprocess


class MyDataset(Dataset):
    """My dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        """ Initialize the dataset. """
        self.data_path = raw_data_path
        self.data = pd.read_csv(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """Return a given sample from the dataset.
        
        Args:
            index (int): Index of the sample to return.
            
        Returns:
            dict: A dictionary containing the sample data.
        """
        row = self.data.iloc[index]
        return {
            "id": row["id"],
            "keyword": row["keyword"],
            "location": row["location"],
            "text": row["text"],
            "target": row["target"],
        }

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder.
        
        Args:
            output_folder (Path): Path to the folder where the processed data will be saved.   
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # fill in missing location values with 'unknown'
        self.data['location'] = self.data['location'].fillna('unknown')

        # Clean 'text' in the dataset by lowercasing, removing:
        # - URLs, hashtags, mentions, special characters, and numbers
        def clean_text(text: str) -> str:
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+",
                          "", text, flags=re.MULTILINE)
            text = re.sub(r"\@\w+|\#", "", text)
            text = re.sub(r"[^A-Za-z]+", " ", text)
            text = re.sub(r"\b\d+\b", "", text)
            return text

        self.data['text'] = self.data['text'].apply(clean_text)

        # Tokenize the cleaned text data
        # bert-base-uncased or BERTweet tokenizer ?
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Combine location and text when tokenizing?
        tokenized_data = tokenizer(
            list(self.data['location'] + " | " + self.data["text"]),
            padding="longest",
            truncation=True,
            # max_length=128, # BERT max length (not sure how long it should be)
            return_tensors="pt",  # PyTorch tensors
        )

        # Extract tensors
        # Tokenized input (101 indicates start of sentence, 102 end of sentence)
        input_ids = tokenized_data["input_ids"]
        # Attention mask (0 for padding, 1 for non-padding)
        attention_mask = tokenized_data["attention_mask"]
        # Target values 1 for disaster-related, 0 for not
        targets = torch.tensor(self.data['target'].values, dtype=torch.long)

        # Save tensors
        torch.save(input_ids, output_folder / "input_ids.pt")
        torch.save(attention_mask, output_folder / "attention_mask.pt")
        torch.save(targets, output_folder / "targets.pt")


def download_dataset(bucket_name: str = "d_tweets", blob_name: str = "tweets.csv", raw_data_dir: Path = Path("data/raw")) -> None:
    """Download dataset from Google Cloud Storage.
    
    Args: bucket_name (str): Name of the bucket in GCS. 
            blob_name (str): Name of the blob to download.
            raw_data_dir (Path): Path to the directory where the raw data will be saved.    
    """
    # Create directory for raw data if it doesn't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the GCS client
    client = storage.Client()

    # Access the bucket and blob
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Local file path to save the downloaded file
    destination_file = raw_data_dir / blob_name
    
    # Download the blob to the destination
    blob.download_to_filename(destination_file)
    print(
        f"Downloaded {blob_name} from bucket {bucket_name} to {destination_file}")



def preprocess(raw_data_path: str = "data/raw/tweets.csv", output_folder: str = "data/processed") -> None:
    """Preprocess the raw data and save it to the output folder.

    Args:
        raw_data_path (str, optional): _description_. Defaults to "data/raw/tweets.csv".
        output_folder (str, optional): _description_. Defaults to "data/processed".
    """
    
    print("Preprocessing data...")
    raw_data_dir = Path(raw_data_path)
    if not raw_data_dir.exists():
        print("Raw data not found, downloading...")
        download_dataset()

    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

    
def tweets() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for tweets.
    
    Returns: 
        Train and test datasets in the form of torch.utils.data.Dataset.
    """
    input_ids = torch.load("data/processed/input_ids.pt")
    attention_mask = torch.load("data/processed/attention_mask.pt")
    targets = torch.load("data/processed/targets.pt")
    print(input_ids.shape, attention_mask.shape, targets.shape)
    # Split the data into training and validation sets
    input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, targets_train, targets_test = train_test_split(
        input_ids, attention_mask, targets, test_size=0.2, random_state=42)

    train_set = torch.utils.data.TensorDataset(
        input_ids_train, attention_mask_train, targets_train)
    test_set = torch.utils.data.TensorDataset(
        input_ids_test, attention_mask_test, targets_test)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
