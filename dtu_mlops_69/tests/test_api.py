from src.mlops_project.api import predict_deployed_api


def test_api(input_data: str = "There's a massive earthquake!", location: str = "California") -> None:
    response = predict_deployed_api(input_data, location)

    # Check if the response status code is 200 (ok)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"

    # Ensure the response contains JSON data
    try:
        prediction = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Check format
    assert prediction is not None, "Response is None"

    assert "prediction" in prediction, "Prediction key not found in response"
    assert prediction["prediction"][0] == 0 or prediction["prediction"][0] == 1, "Invalid prediction value"
