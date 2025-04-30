# similar to preprocess_users_ratings.py, but for one user
import numpy as np
import pandas as pd

def preprocess_individual(merged_df, user_id, cuisines_file='cuisines.txt'):
   """
   Given merged_df (reviews merged with business info) and a specific user_id,
   return the user's average ratings across each cuisine type.
   
   Args:
      merged_df (pd.DataFrame): DataFrame with user reviews merged with business categories.
      user_id (str): The target user's user_id.
      cuisines_file (str): Path to the cuisines.txt file listing cuisines (one per line).
   
   Returns:
      pd.Series: Average rating per cuisine for the user (index = cuisines).
   """

   # load cuisines list
   delimiter = '\n'
   with open(cuisines_file, 'r', encoding='utf-8') as file:
      cuisines = file.read().split(delimiter)
   
   # create dictionary to hold cuisine-specific restaurant dfs
   df_cuisines = {}
   business_df = merged_df[['business_id', 'categories']].drop_duplicates()

   for cuisine in cuisines:
      df_cuisines[cuisine] = business_df[business_df['categories'].str.contains(cuisine, case=False, na=False)]

   # initialize user vector
   user_vector = []

   # filter reviews by this specific user
   user_reviews = merged_df[merged_df['user_id'] == user_id]

   # for each cuisine, compute average rating
   for cuisine in cuisines:
      businesses_in_cuisine = df_cuisines[cuisine]['business_id'].unique()
      user_reviews_in_cuisine = user_reviews[user_reviews['business_id'].isin(businesses_in_cuisine)]

      if not user_reviews_in_cuisine.empty:
         avg_rating = user_reviews_in_cuisine['stars'].mean()
      else:
         avg_rating = np.nan
      
      user_vector.append(avg_rating)
   
   # user_id becomes the index, cuisines is the columns
   user_cuisine_profile = pd.DataFrame({cuisine: [rating] for cuisine, rating in zip(cuisines, user_vector)}, index=[user_id])
   user_cuisine_profile.index.name = 'user_id'

   return user_cuisine_profile

if __name__ == '__main__':
   # load in df from .pkl file
   df = pd.read_pickle('data/reviews.pkl')
   business_df = pd.read_pickle('data/business.pkl')

   # preprocess business & categories
   merged_df = pd.merge(df, business_df[['business_id', 'categories']], on='business_id', how='left')

   top_users_df = pd.read_pickle('data/user_ratings.pkl') # top 10 users with most reviews
   top_user_ids = top_users_df.index.tolist()
   target_user_id = top_user_ids[0]

   print(preprocess_individual(merged_df, target_user_id))