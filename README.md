# dtu_mlops_69
# Disaster Tweets Classification

### Group Members
- [x] s185034 - Andreas Patscheider 
- [x] s200621 - Mads Rumle Nordstrøm
- [x] s194045 - Niklas August Kjølbro
- [x] s203953 - Rasmus Porsgaard
- [x] s240398 - Sofus Erdal

# Overall goal of the project
In this project we aim to implement NLP to classify tweets as either disaster or non-disaster tweets. <br>
We will use the Disaster-Tweets dataset from Kaggle to train our model upon and the model will be trained using a transformer model, and we will use the Huggingface library to implement the model. <br>
The model will be trained using the PyTorch framework and the training will be done using PyTorch Lightning abd to manage our hyperparameters and configurations we will use Hydra. <br>
Finally we will use wandb to log training progress and other important metrics/artifacts in our code and  will also use Docker to containerize our code. <br>
The overall purpose of the project is to implement a model that can classify tweets as either disaster or non-disaster tweets and put this model into production. <br>
With this project we aim to learn how to implement NLP models and put them into production using the frameworks mentioned above. <br>

# What frameworks (may change)
As mentioned in the project description, we will use the following frameworks:
- PyTorch
- PyTorch Lightning
- Huggingface (transformers)
- Hydra
- wandb
- Docker

# data: 
The data we will use for this project is the [Disaster-Tweets Dataset](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets).
The data <italic> contains over 11,000 tweets associated with disaster keywords like “crash”, “quarantine”, and “bush fires” as well as the location and keyword itself. </italic> The dataset is labeled with 1 for disaster tweets and 0 for non-disaster tweets.
With this data we are able to train our model to classify tweets as either disaster or non-disaster tweets. 


# What models 
As we will be dealing with NLP, we will use a transformer model to classify the tweets. We will use the Huggingface library to implement the model. <br>
Wihtin the Huggingface library numerous BERT-models and <italic> the first public large-scale pre-trained language model for English Tweets </italic>, the [BERTweet](https://huggingface.co/docs/transformers/model_doc/bertweet) will definitely be considered. <br>
Additionally we will experiment with other version of the BERT-model to see which one performs the best on our dataset. <br>


Check the checklist.md file for the project progress
