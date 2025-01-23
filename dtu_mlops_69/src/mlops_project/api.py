import requests
import typer
from datetime import datetime, timezone

app = typer.Typer()

# The deployed cloud function URL
API_URL = "https://europe-west3-dtumlops-448112.cloudfunctions.net/tweet_predict"

# Function to send a POST request to the deployed API


def predict_deployed_api(input_data: str, location: str) -> requests.Response:
    now = str(datetime.now(tz=timezone.utc))
    test_data = {
        "input_data": [input_data],
        "location": location
    }

    # Send POST request to the deployed API
    response = requests.post(API_URL, json=test_data)

    if response.status_code == 200:
        # Parse and print the prediction result
        prediction = response.json()
        print(f"Prediction: {prediction}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

    # Add to database
    add_to_database(now, input_data, location, prediction["prediction"][0])

    # Return the response
    return response

# Add to database


def add_to_database(now: str, input_data: str, location: str, prediction: str) -> None:
    """Simple function to add prediction to database."""
    with open("prediction_api/prediction_database.csv", "a") as file:
        file.write(f"{now}, {location}, {input_data},{prediction}\n")

# Typer command to send a request to the deployed API


@app.command()
def send_disaster_request(text: str, location: str):
    """Take input data (tweet text) and location, then send to the deployed API."""
    print(f"Sending request with tweet: {text} and location: {location}")
    predict_deployed_api(text, location)


if __name__ == "__main__":
    # Test it out:
    # python src/mlops_project/api.py send-disaster-request text="There's a massive earthquake!" location="California""
    app()
