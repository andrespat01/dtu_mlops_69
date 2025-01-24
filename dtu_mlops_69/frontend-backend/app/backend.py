# from contextlib import asynccontextmanager

# import torch
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


# from fastapi import FastAPI
# import regex as re

# app = FastAPI()


# from http import HTTPStatus


# from contextlib import asynccontextmanager

# import torch
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load and clean up model on startup and shutdown."""
#     global model, feature_extractor, tokenizer, device, gen_kwargs
#     print("Loading model")
#     model = VisionEncoderDecoderModel.from_pretrained(
#         "nlpconnect/vit-gpt2-image-captioning"
#     )
#     feature_extractor = ViTFeatureExtractor.from_pretrained(
#         "nlpconnect/vit-gpt2-image-captioning"
#     )
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

#     yield

#     print("Cleaning up")
#     del model, feature_extractor, tokenizer, device, gen_kwargs


# app = FastAPI(lifespan=lifespan)


# @app.post("/caption/")
# async def caption(data: UploadFile = File(...)):
#     """Generate a caption for an image."""
#     i_image = Image.open(data.file)
#     if i_image.mode != "RGB":
#         i_image = i_image.convert(mode="RGB")

#     pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     return [pred.strip() for pred in preds]

from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
import re
import io

# Constants
BUCKET_NAME = "tweets-models"
MODEL_FILE = "model.pth"

# Initialize FastAPI app
app = FastAPI()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Model Definition
class DisasterTweetModel(nn.Module):
    def __init__(self):
        super(DisasterTweetModel, self).__init__()
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


# Utility functions
def clean_text(text: str) -> str:
    """Clean the tweet text by removing URLs, hashtags, mentions, and special characters."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z]+", " ", text)
    text = re.sub(r"\b\d+\b", "", text)
    return text


def load_model_from_gcs(bucket_name: str, model_file: str):
    """Load the model from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(model_file)
    model_data = blob.download_as_string()

    model_buffer = io.BytesIO(model_data)
    model = DisasterTweetModel()
    model.load_state_dict(torch.load(model_buffer))
    model.eval()
    return model


model_loaded = False  # Global flag to track model readiness


@app.on_event("startup")
async def load_model():
    global model, model_loaded
    print("Loading model...")
    model = load_model_from_gcs(BUCKET_NAME, MODEL_FILE)
    model_loaded = True  # Set the flag to True once the model is loaded
    print("Model loaded.")


@app.get("/health")
async def health_check():
    if model_loaded:
        return {"status": "ok", "message": "Model is loaded and API is ready."}
    else:
        return {"status": "loading", "message": "Model is still loading."}


# create root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Disaster Tweet Classifier!"}


class InferenceRequest(BaseModel):
    input_data: list[str]
    location: Optional[str] = None


class InferenceResponse(BaseModel):
    prediction: list[int]


@app.post("/predict/")
async def predict(request: InferenceRequest):
    global model  # Ensure you're referencing the global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    input_data = request.input_data
    location = clean_text(request.location) if request.location else "unknown"

    if not input_data:
        raise HTTPException(status_code=400, detail="No input data provided.")

    # Clean and format text
    cleaned_text = f"{location} | {clean_text(input_data[0])}"
    inputs = tokenizer(
        cleaned_text, padding=True, truncation=True, return_tensors="pt", max_length=128
    )
    sent_id = inputs["input_ids"]
    mask = inputs["attention_mask"]

    # Make prediction
    with torch.no_grad():
        outputs = model(sent_id, mask)

    prediction = outputs.argmax(dim=1).tolist()[0]  # Extract the first prediction

    # Map prediction to descriptive string
    if prediction == 1:
        message = "This is a disaster tweet."
    else:
        message = "This is not a disaster tweet."

    return {"prediction": prediction, "message": message}
