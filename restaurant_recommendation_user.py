import pandas as pd
from scaled_cos_similarity import scaled_cos_sim

# load data
reviews_df = pd.read_pickle('reviews.pkl')
business_df = pd.read_pickle('business.pkl')
user_ratings_df = pd.read_pickle('user_ratings.pkl')  # your user preference profiles
top_users_df = pd.read_pickle('user_ratings.pkl') # top 10 users with most reviews
top_business_df = pd.read_csv('top_reviewed_restaurants.csv') # top 10 most reviewed restaurants

# turn into lists
top_user_ids = top_users_df.index.tolist()
top_business_ids = top_business_df[top_business_df.columns[0]].tolist()

# arbitary, select any top user and top restaurant for testing
target_business_id = top_business_ids[0]
target_user_id = top_user_ids[0]
target_user_vector = user_ratings_df.loc[target_user_id].fillna(0).astype(float).tolist() # convert to list

# snippet to test functionality; by selecting a business that the target user has reviewed, we expect the same user to appear in sim index with sim 1.0
# user_reviews = reviews_df[reviews_df['user_id'] == target_user_id]
# target_business_id = user_reviews.iloc[0]['business_id']

# find all users that reviewed the target restaurant
reviewed_users = reviews_df[reviews_df['business_id'] == target_business_id]['user_id'].unique()

# calculate scaled cosine similarity with each reviewed user
similarities = [] # save here

for user_id in reviewed_users:
   if user_id in user_ratings_df.index:
      other_user_vector = user_ratings_df.loc[user_id].fillna(0).astype(float).tolist()
      sim = scaled_cos_sim(target_user_vector, other_user_vector)
      similarities.append((user_id, sim))

# find best matches w/ highest similarities
if similarities:
   similarities.sort(key=lambda x: x[1], reverse=True)
   top_matches = similarities[:3] # takes the top 3 matches

print(f"Top users most similar to {target_user_id} who reviewed {target_business_id}:")
if similarities: # if non-empty
   for idx, (user_id, sim_score) in enumerate(top_matches, 1):
      print(f"User {idx}: {user_id}, similarity: {sim_score:.4f}")
else:
   print("No matching reviewers found.")
