import os
import requests
import streamlit as st
from google.cloud import run_v2


# Function to get the backend URL
@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/dtumlops-448112/locations/europe-west3"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_tweet(tweet: str, location: str, backend_url: str):
    """Send the tweet and location to the backend for classification."""
    payload = {"input_data": [tweet], "location": location}
    response = requests.post(f"{backend_url}/predict/", json=payload, timeout=100)

    if response.status_code == 200:
        return response.json()
    return None


def main():
    """Main function for the Streamlit app."""
    backend_url = get_backend_url()
    if not backend_url:
        st.error("Backend service not found. Please check your configuration.")
        return

    st.title("Disaster Tweet Classifier")

    # Input fields
    tweet = st.text_area("Enter your tweet", placeholder="Type your tweet here...")
    location = st.text_input(
        "Enter the location (optional)", placeholder="Type location here..."
    )

    # Submit button
    if st.button("Submit"):
        if not tweet.strip():
            st.error("Please enter a tweet to classify.")
        else:
            # Send the input to the backend
            result = classify_tweet(tweet, location, backend_url)

            if result:
                st.success("Prediction result:")
                st.write(result["message"])  # Display descriptive prediction message
                st.json(result)  # Show raw JSON response for debugging
            else:
                st.error("Failed to get a prediction. Please try again.")


if __name__ == "__main__":
    main()
