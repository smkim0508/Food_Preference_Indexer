# knn rating predictor

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

def predict_rating_knn(
target_user_id: str,
target_business_id: str,
user_ratings_df: pd.DataFrame,
reviews_df: pd.DataFrame,
k: int = 5,
metric: str = 'cosine'
) -> Tuple[float, List[str]]:
   """
   Predict the rating target_user_id would give to target_business_id
   by finding the k nearest neighbors (in taste-space) among users
   who have reviewed that business, then averaging *their* ratings.

   Returns:
   - predicted_rating (float)
   - neighbor_ids (List[str])  ← the top-k neighbor user IDs
   """
   # prepare the ratings matrix
   X = user_ratings_df.fillna(0).astype(float)

   if target_user_id not in X.index:
      raise KeyError(f"Target user {target_user_id} not in user_ratings_df")

   # find every user that reviewed target business
   reviewers = (
      reviews_df[reviews_df['business_id'] == target_business_id]
      ['user_id']
      .unique()
      .tolist()
   )
   candidates = [u for u in reviewers if u in X.index and u != target_user_id]
   if not candidates:
      return np.nan, []

   # build feature matrices
   X_cand = X.loc[candidates].values
   x_target = X.loc[[target_user_id]].values

   # fit KNN
   n_neighbors = min(k, len(candidates))
   nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
   nn.fit(X_cand)

   # query neighbors
   distances, idxs = nn.kneighbors(x_target, return_distance=True)
   neighbor_idxs = idxs[0]

   # 6) map back to user_ids
   neighbor_ids = [candidates[i] for i in neighbor_idxs]

   # 7) collect their actual ratings
   neigh_ratings = reviews_df[
      (reviews_df['business_id'] == target_business_id) &
      (reviews_df['user_id'].isin(neighbor_ids))
   ]['stars']

   pred = neigh_ratings.mean() if not neigh_ratings.empty else np.nan
   return pred, neighbor_ids

# testing
if __name__ == "__main__":
   # load data for testing
   user_ratings_df    = pd.read_pickle('data/user_ratings_gt_300.pkl')
   reviews_df         = pd.read_pickle('data/reviews.pkl')
   top_business_df    = pd.read_csv('top_reviewed_restaurants.csv')
   business_ids       = top_business_df[top_business_df.columns[0]].tolist()

   # pick sample inputs
   top_users_df       = pd.read_pickle('data/user_ratings.pkl')
   target_user_id     = top_users_df.index.tolist()[1]
   target_business_id = business_ids[0]
   k = int(input("Number of neighbors k: ").strip())

   # run prediction
   pred, neighbors = predict_rating_knn(
      target_user_id,
      target_business_id,
      user_ratings_df,
      reviews_df,
      k=k,
      metric='cosine'
   )

   # check output
   if not neighbors:
      print("Not enough neighbors to predict.")
   else:
      print(f"Top {len(neighbors)} neighbors used: {neighbors}")
      if np.isnan(pred):
         print("Those neighbors didn't have a rating for that business.")
      else:
         print(f"Predicted rating for user {target_user_id} on {target_business_id}: {pred:.2f}")
