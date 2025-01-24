from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
import os
import pytest
import torch

# Append the backend folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frontend-backend"))
from app.backend import app  # Import the backend app

import time

client = TestClient(app)
model = None


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Disaster Tweet Classifier!"}


def test_predict_no_input_data():
    """Test the /predict/ endpoint with no input data."""
    payload = {}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Update expected status code
    assert "detail" in response.json()


# def wait_for_model_ready(timeout=360):
#     """Wait for the model to load by polling the /health endpoint."""
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         response = client.get("/health")
#         if response.status_code == 200 and response.json().get("status") == "ok":
#             return True
#         time.sleep(1)
#     raise TimeoutError("Model did not load within the timeout period.")


# def test_predict_disaster_tweet():
#     """Test the /predict/ endpoint with a disaster tweet."""
#     # Wait for the model to load
#     wait_for_model_ready()

#     payload = {
#         "input_data": ["Floods have devastated the city and people need urgent help."],
#         "location": "Los Angeles",
#     }
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "prediction" in response_data
#     assert "message" in response_data
#     assert response_data["prediction"] == 1
#     assert response_data["message"] == "This is a disaster tweet."


# def test_predict_non_disaster_tweet():
#     """Test the /predict/ endpoint with a non-disaster tweet."""

#     # Wait for the model to load
#     wait_for_model_ready()
#     payload = {
#         "input_data": ["What a beautiful sunny day!"],
#         "location": "San Francisco",
#     }
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "prediction" in response_data
#     assert "message" in response_data
#     assert response_data["prediction"] == 0
#     assert response_data["message"] == "This is not a disaster tweet."


# def test_predict_multiple_tweets():
#     """Test the /predict/ endpoint with multiple tweets (only the first is processed)."""

#     # Wait for the model to load
#     wait_for_model_ready()
#     payload = {
#         "input_data": [
#             "Earthquake reported near the coast. Stay safe!",
#             "This is just a regular tweet about coffee.",
#         ],
#         "location": "California",
#     }
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "prediction" in response_data
#     assert "message" in response_data
#     assert response_data["prediction"] == 1  # Only the first tweet is processed


# def test_predict_invalid_input():
#     """Test the /predict/ endpoint with invalid input data."""

#     # Wait for the model to load
#     wait_for_model_ready()
#     payload = {"input_data": [12345], "location": "New York"}  # Invalid data type
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 422  # Unprocessable Entity
#     assert "detail" in response.json()


# def test_predict_empty_input():
#     """Test the /predict/ endpoint with empty input."""

#     # Wait for the model to load
#     wait_for_model_ready()
#     payload = {"input_data": [""], "location": ""}
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "prediction" in response_data
#     assert "message" in response_data
#     assert response_data["prediction"] == 0  # No meaningful content
#     assert response_data["message"] == "This is not a disaster tweet."


# def test_predict_large_input():
#     """Test the /predict/ endpoint with a large input."""

#     # Wait for the model to load
#     wait_for_model_ready()
#     large_text = " ".join(["disaster"] * 1000)  # Simulate a very large input
#     payload = {"input_data": [large_text], "location": "Unknown"}
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "prediction" in response_data
#     assert "message" in response_data
