import numpy as np
import pandas as pd

# scaled cosine similarity, accounting for magnitude
def scaled_cos_sim(a, b):
   norm_a = np.linalg.norm(a)
   norm_b = np.linalg.norm(b)
   cos_sim = np.dot(a, b) / (norm_a * norm_b) # calculate cosine similarity value
   mag_ratio = min(norm_a, norm_b) / max(norm_a, norm_b) # calculate ratio of magnitudes to scale
   return cos_sim * mag_ratio  # reduces score when magnitudes differ

# load in top 10 user data
all_user_ratings_df = pd.read_pickle('user_ratings.pkl') # many users
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
      similarities.append((idx, sim))
   
   # sort by the most similar users
   similarities.sort(key=lambda x: x[1], reverse=True)

   # pick top 3 matches and save
   top_3 = similarities[:3]
   top_matches[target_idx] = top_3

# print results
for target_user, matches in top_matches.items():
   print(f"Target user {target_user} top matches:")
   for user_idx, sim in matches:
      print(f"\tUser {user_idx} with similarity {sim:.4f}")