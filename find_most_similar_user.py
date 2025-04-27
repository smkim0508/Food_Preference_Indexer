import numpy as np
import pandas as pd
from scaled_cos_similarity import scaled_cos_sim # import function

# load in all users data
# all_user_ratings_df = pd.read_pickle('user_ratings.pkl') # to load top 10 users with most reviews
all_user_ratings_df = pd.read_pickle('user_ratings_gt_300.pkl') # to load all users with more than 100 reviews
all_user_ids = all_user_ratings_df.index.tolist() # also save the user_ids to map later
# load in the target user to compare with
target_user_ratings_df = pd.read_pickle('dummy_ratings.pkl') # few users

# handle NaN with 0
all_user_ratings_df = all_user_ratings_df.fillna(0)
target_user_ratings_df = target_user_ratings_df.fillna(0)

# convert all values to float just in case
all_user_ratings_df = all_user_ratings_df.astype(float)
target_user_ratings_df = target_user_ratings_df.astype(float)

# convert df to 2D lists
all_user_vectors = all_user_ratings_df.astype(float).apply(lambda row: row.tolist(), axis=1).tolist()
target_user_vectors = target_user_ratings_df.astype(float).apply(lambda row: row.tolist(), axis=1).tolist()

# store top matches
top_matches = {}

# loop through each target user
for target_idx, target_vec in enumerate(target_user_vectors):
   similarities = []

   # loop through all users
   for idx, user_vec in enumerate(all_user_vectors):
      sim = scaled_cos_sim(target_vec, user_vec)
      # keep track of all similarities
      similarities.append((idx, all_user_ids[idx], sim))
   
   # sort by the most similar users
   similarities.sort(key=lambda x: x[2], reverse=True)

   # pick top 3 matches and save
   top_3 = similarities[:3]
   top_matches[target_idx] = top_3

# print results
for target_user, matches in top_matches.items():
   print(f"Target user {target_user} top matches:")
   for idx, user_id, sim in matches:
      print(f"\tUser {idx}: {user_id} with similarity {sim:.4f}")