from fastapi import FastAPI

from fastapi.testclient import TestClient

import sys
import os

# append the following folder to sys.path: dtu_mlops_69/frontend-backend/app/backend.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frontend-backend"))
from app.backend import app
import pytest
import torch

client = TestClient(app)
model = None


# @pytest.fixture
# def mock_model(monkeypatch):
#     """Mock the global model for testing."""

#     class MockModel:
#         def eval(self):
#             pass

#         def __call__(self, input_ids, attention_mask):
#             # Return a tensor mimicking the model's output
#             return torch.tensor([[0.2, 0.8]])  # Simulate prediction for class 1

#     # Mock the global model in app.backend
#     mock_instance = MockModel()
#     monkeypatch.setattr("app.backend", "model", mock_instance)


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Disaster Tweet Classifier!"}


# def test_predict_disaster_tweet():
#     """Test the /predict/ endpoint for a disaster tweet."""
#     payload = {
#         "input_data": ["There is a severe storm coming to the city!"],
#         "location": "New York",
#     }
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     assert response.json()["prediction"] == 1
#     assert response.json()["message"] == "This is a disaster tweet."


# def test_predict_non_disaster_tweet():
#     """Test the /predict/ endpoint for a non-disaster tweet."""
#     payload = {"input_data": ["I am having a wonderful day!"], "location": "California"}
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     assert response.json()["prediction"] == 0
#     assert response.json()["message"] == "This is not a disaster tweet."


def test_predict_no_input_data():
    """Test the /predict/ endpoint with no input data."""
    payload = {}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Update expected status code
    assert "detail" in response.json()


# def test_predict_missing_location():
#     """Test the /predict/ endpoint with missing location."""
#     payload = {"input_data": ["There is flooding in the area."]}
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == 200
#     assert response.json()["prediction"] == 1
#     assert response.json()["message"] == "This is a disaster tweet."
