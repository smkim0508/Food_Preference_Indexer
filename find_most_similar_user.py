import numpy as np
import pandas as pd
from scaled_cos_similarity import scaled_cos_sim # import function

def find_top_matches(all_users_path, target_user_path, top_n=3):
   
   # load in all users data
   # all_user_ratings_df = pd.read_pickle('user_ratings.pkl') # to load top 10 users with most reviews
   all_user_ratings_df = pd.read_pickle(all_users_path) # to load all users with more than 100 reviews
   all_user_ids = all_user_ratings_df.index.tolist() # also save the user_ids to map later
   # load in the target user to compare with
   target_user_ratings_df = pd.read_pickle(target_user_path)

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
   top_matches = ()
   similarities = []

   # loop through all users
   for idx, user_vec in enumerate(all_user_vectors):
      sim = scaled_cos_sim(target_user_vectors[0], user_vec) # target_user_vectors[0] if given as a 2Dlist of target users
      # keep track of all similarities
      similarities.append((idx, all_user_ids[idx], sim))
   
   # sort by the most similar users
   similarities.sort(key=lambda x: x[2], reverse=True)

   # pick top_n matches and save
   top_matches = similarities[:top_n]

   # print results
   for idx, user_id, sim in top_matches:
      print(f"\tUser {idx}: {user_id} with similarity {sim:.4f}")

# to test the script
if __name__ == "__main__":
   find_top_matches('user_ratings_gt_300.pkl','dummy_ratings.pkl',3)