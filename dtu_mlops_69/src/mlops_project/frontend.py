import streamlit as st
from api import predict_deployed_api

# Function to simulate prediction
def predict(tweet: str, location: str) -> int:
    response = predict_deployed_api(tweet, location)
    prediction = response.json()
    return prediction["prediction"][0]

def main():
    # Streamlit app
    st.title("Disaster tweet prediction!")

    # Input boxes for tweet and location
    tweet = st.text_area("Enter tweet:", "")
    location = st.text_input("Enter location:", "")

    # Button
    if st.button("Get Prediction"):
        if not tweet:
            st.error("Tweet box cannot be empty")
        else:
            # prediction
            prediction = predict(tweet, location)
            
            if prediction == 0:
                st.success("Our model predicts that this was NOT about a real disaster.")
            else:
                st.success("Our model predicts that this tweet was about a real disaster.")

if __name__ == "__main__":
    # To run:
    # streamlit run src/mlops_project/frontend.py
    main()
