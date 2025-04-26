import numpy as np
import pandas as pd

# load in df from .pkl file
df = pd.read_pickle('reviews.pkl')
business_df = pd.read_pickle('business.pkl')

# preprocess business & categories
merged_df = pd.merge(df, business_df[['business_id', 'categories']], on='business_id', how='left')

sorted_df = merged_df.sort_values(by=['business_id', 'categories'])

# preprocess user_ids
grouped_users = df.groupby('user_id')['business_id'].apply(list).reset_index()
grouped_users['review_count'] = grouped_users['business_id'].apply(len) # add a column for how many reviews each user wrote
users_sorted = grouped_users.sort_values(by=['review_count'], ascending=False) # sort df to place users with most reviews at the top

top_users = users_sorted.head(10) #take top users with the most number of reviews

# df_cuisines
# load in cuisines.txt to parse data
df_cuisines = {}
cuisines = []
delimiter = '\n'

# use short cuisines list for testing
with open('cuisines.txt', 'r', encoding='utf-8') as file:
   cuisines = file.read().split(delimiter)

# creating a dict where each keyword = cuisine, each value = pandas df that contains sorted_df filtered by each cuisine category
for cuisine in cuisines:
   keyword = cuisine
   # create separate df for each cuisine in cuisines list
   df_cuisines[keyword] = business_df[business_df['categories'].str.contains(keyword, case=False, na=False)]


# print(df['business_id'].head())
# print(df_cuisines['Korean']['business_id'].head())


# initialize the final dictionary
user_ratings = {}

# go through each user
for user_id in top_users['user_id']:
   user_vector = []  # one value per cuisine
   
   # filter all reviews by this user
   user_reviews = sorted_df[sorted_df['user_id'] == user_id]
   # print(user_reviews)
   
   for cuisine in cuisines:
      # get business_ids for this cuisine
      businesses_in_cuisine = df_cuisines[cuisine]['business_id'].unique()
      
      # find user's reviews for businesses in this cuisine
      user_reviews_in_cuisine = user_reviews[user_reviews['business_id'].isin(businesses_in_cuisine)]
   
      if not user_reviews_in_cuisine.empty:
         avg_rating = user_reviews_in_cuisine['stars'].mean()
      else:
         avg_rating = np.nan
      
      user_vector.append(avg_rating)
   user_ratings[user_id] = user_vector

# convert user_profiles to a df, then save values
user_ratings_df = pd.DataFrame.from_dict(user_ratings, orient='index', columns=cuisines)

# save as .csv for easy visual check
user_ratings_df.to_csv('user_ratings.csv', index=True)
# save as .pkl for python efficiency
user_ratings_df.to_pickle('user_ratings.pkl')

