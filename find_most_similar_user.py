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
best_matches = {}

# loop through each target user
for target_idx, target_vec in enumerate(target_user_vectors):
   best_sim = -1  # start with lowest similarity
   best_user_idx = None
   # loop through all users
   for idx, user_vec in enumerate(all_user_vectors):
      if (idx == 1): print(user_vec)
      sim = scaled_cos_sim(target_vec, user_vec)
      # print(sim)
      if sim > best_sim:
         best_sim = sim
         best_user_idx = idx
   
   # save the best matching user for this target user
   best_matches[target_idx] = (best_user_idx, best_sim)

# check results
for target_user, (best_user, similarity) in best_matches.items():
    print(f"Target user {target_user} best matches with {best_user} (similarity = {similarity:.4f})")