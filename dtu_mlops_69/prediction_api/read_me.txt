Create the API instance:
# Zone = Germany (Frankfurt)
# Memory usage = 2GB
# CPU = 2

Ikke prøv og kør denne kode 
(den vil lave en nyt instance eller skrive over den nuværende)

gcloud functions deploy tweet_predict \
  --runtime python311 \
  --trigger-http \
  --source prediction_api/api \
  --entry-point disaster_tweet_classifier \
  --memory 2GB \
  --cpu 2 \
  --region europe-west3

# Make a prediction:
# Input to the model is:
# input = 'location | text' 
# if location not provided then default to 'unknown'

curl -X POST \
  https://europe-west3-dtumlops-448112.cloudfunctions.net/tweet_predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": ["Lord Jesus, your love brings freedom and pardon. Fill me with your Holy Spirit and set my heart ablaze with your l… https://t.co/VlTznnPNi8"], "location": "OC"}'

# Without location:
curl -X POST \
  https://europe-west3-dtumlops-448112.cloudfunctions.net/tweet_predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": ["Lord Jesus, your love brings freedom and pardon. Fill me with your Holy Spirit and set my heart ablaze with your l… https://t.co/VlTznnPNi8"]}'

