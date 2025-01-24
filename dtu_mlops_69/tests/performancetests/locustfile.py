import random
from locust import HttpUser, between, task


class DisasterTweetUser(HttpUser):
    """Simulates user behavior for testing the Disaster Tweet Classifier API."""

    # Simulate a wait time between requests to mimic real-world user behavior
    wait_time = between(1, 3)

    @task(1)
    def get_root(self) -> None:
        """Simulates a user accessing the root endpoint."""
        self.client.get("/")

    @task(5)
    def post_predict_disaster_tweet(self) -> None:
        """Simulates a user posting a disaster tweet for classification."""
        payload = {
            "input_data": ["There is a severe storm coming to the city!"],
            "location": "New York",
        }
        self.client.post("/predict/", json=payload)

    @task(3)
    def post_predict_non_disaster_tweet(self) -> None:
        """Simulates a user posting a non-disaster tweet for classification."""
        payload = {
            "input_data": ["I am having a wonderful day!"],
            "location": "California",
        }
        self.client.post("/predict/", json=payload)

    @task(2)
    def post_predict_missing_location(self) -> None:
        """Simulates a user posting a tweet with no location provided."""
        payload = {"input_data": ["There is flooding in the area."]}
        self.client.post("/predict/", json=payload)
