# Python-Machine-Learning

In these two projects, I have leveraged the power of Natural Language Processing (NLP) and Recurrent Neural Network (RNN) models to accurately predict categorical features within two datasets.

## Natural Language Processing (NLP) Project:
Analyzing a dataset of Yelp reviews from [Kraggle](https://www.kaggle.com/c/yelp-recsys-2013), I used exploratory data analysis and implemented an NLP model using the Python sklearn library in order to:
* Graph distributions of each Yelp review by stars (1-5) and text length
* Find if any strong correlations lie in the data using a heatmap
* Separate the data into train/test splits & use CountVectorizer function to remove any stopwords from the reviews
* Implement and train a Pipeline model that uses a MultinomialNB classifier and Tf-idf transformer
* Create a classification report & confusion matrix to determine the accuracy of the model

The model had an accuracy f1-score of 0.81, meaning, given a text Yelp review, the model had approximately an 81% chance of guessing right whether it was a 5-star or 1-star review.

