# Food_Preference_Indexer

Personalized Restaurant Recommendation System, originally developed in part for final project in SML312: Independent Projects in Data Science taught by Professor Jonathan Hanke at Princeton University. 

The recommendations are made with two main objectives identified below:

1. Given a user, identifying which other user in the database has the most similar taste preferences.

2. Given a restaurant and a user, predicting what rating the user will give that restaurant from 1-5.

I utilize a number of data science modeling methods (e.g. KNN, Multivariate Regression with Ridge and Lasso Regularization, Random Forest Classification) as well as a scaled cosine similarity-based static recommendation system. The scaled cosine similarity approach is used to return recommendations for users in an external iOS app "Restaurant Repo" (can be found in my GitHub page as well), using a FLASK server connection using the function defined in 'suggest_modular.py'. Currently, the location filtering has been commented out due to a lack of user data. 

I experimented using the Yelp Dataset, which can be found at: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_business.json
