import torch
import functions_framework
from google.cloud import storage
from transformers import AutoModel, AutoTokenizer
from torch import nn
import re
import io

BUCKET_NAME = "tweets-models"
MODEL_FILE = "model.pth"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Model:


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)  # Binary classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = self.fc1(outputs.last_hidden_state[:, 0, :])  # Use the CLS token
        x = torch.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


# Load the model from Google Cloud Storage
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
model_data = blob.download_as_string()

# Wrap the model data in a BytesIO buffer to make it seekable
model_buffer = io.BytesIO(model_data)

# Initialize the model and load the state_dict
model = model()
model.load_state_dict(torch.load(model_buffer))
model.eval()


def clean_text(text: str) -> str:
    """Clean the tweet text by removing URLs, hashtags, mentions, and special characters."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z]+", " ", text)
    text = re.sub(r"\b\d+\b", "", text)
    return text


@functions_framework.http
def disaster_tweet_classifier(request):
    """Classifier function for disaster tweets prediction."""
    request_json = request.get_json()

    if request_json and "input_data" in request_json:
        input_data = request_json["input_data"]

        # Default to location = "unknown" then check if location is provided
        location = "unknown"
        if "location" in request_json and request_json["location"]:
            location = clean_text(request_json["location"])

        # Clean the text
        # Format input of the tweet text to model:
        # input = 'location | text'
        cleaned_text = f"{location} | {clean_text(input_data[0])}"
        inputs = tokenizer(cleaned_text, padding=True,
                           truncation=True, return_tensors="pt", max_length=128)
        sent_id = inputs["input_ids"]
        mask = inputs["attention_mask"]

        # Make prediction
        with torch.no_grad():
            outputs = model(sent_id, mask)

        # Get the prediction (class with highest score)
        prediction = outputs.argmax(dim=1).tolist()

        return {"prediction": prediction}

    return {"error": "No input data provided."}
