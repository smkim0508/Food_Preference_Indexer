# recommender.py

import pandas as pd
from find_most_similar_user import find_top_matches
from preprocess_individual_user import preprocess_individual

def get_user_reviewed_restaurants(user_id: str, reviews_df: pd.DataFrame) -> pd.DataFrame:
   return reviews_df[reviews_df['user_id'] == user_id]

def filter_by_state(restaurants_df: pd.DataFrame, state: str, business_df: pd.DataFrame) -> pd.DataFrame:
   merged = restaurants_df.merge(
      business_df[['business_id', 'state']], on='business_id', how='left'
   )
   return merged[merged['state'] == state]

def filter_by_city(restaurants_df: pd.DataFrame, city: str, business_df: pd.DataFrame) -> pd.DataFrame:
   merged = restaurants_df.merge(
      business_df[['business_id', 'city']], on='business_id', how='left'
   )
   return merged[merged['city'] == city]

def filter_by_cuisine(restaurants_df: pd.DataFrame, cuisine: str, business_df: pd.DataFrame) -> pd.DataFrame:
   merged = restaurants_df.merge(
      business_df[['business_id', 'categories']], on='business_id', how='left'
   )
   return merged[merged['categories'].fillna('').str.contains(cuisine, case=False)]

def sort_by_high_rating(restaurants_df: pd.DataFrame) -> pd.DataFrame:
   return restaurants_df.sort_values(by='stars', ascending=False)

def recommend_restaurants_for_user(
   target_user_id: str,
   all_user_ratings_df: pd.DataFrame,
   reviews_df: pd.DataFrame,
   business_df: pd.DataFrame,
   state: str,
   city: str,
   cuisine: str,
   top_n: int = 5
) -> (pd.DataFrame, str, float):
   """
   Return a Tuple of:
      - DataFrame of recommended restaurants (business_id, name, stars, categories)
      - matched_user_id: the user_id whose reviews we used
      - match_score: similarity value in [0..1]
   for the given target_user_id, filtered by state, city, and cuisine, by looking
   at the top_n most similar users' reviewed restaurants.
   """
   # 1) Build the per-user rating DataFrame for the target user
   merged = reviews_df.merge(
      business_df[['business_id', 'categories']], 
      on='business_id', how='left'
   )
   target_ratings_df = preprocess_individual(merged, target_user_id)

   # target_ratings_df = pd.read_pickle('data/dummy_ratings.pkl') # dummy to test

   # 2) Find the top_n most similar users
   top_matches = find_top_matches(all_user_ratings_df, target_ratings_df, top_n)
   # top_matches is a list of tuples (idx, user_id, similarity)

   # 3) Iterate through those matches until we find some recommendations
   for _, matched_user_id, sim in top_matches:
      user_revs = get_user_reviewed_restaurants(matched_user_id, reviews_df)
      # by_state  = filter_by_state(user_revs,  state,   business_df)
      # by_city   = filter_by_city(by_state,    city,    business_df)
      by_cuisine= filter_by_cuisine(user_revs,  cuisine, business_df)
      sorted_df = sort_by_high_rating(by_cuisine)

      if not sorted_df.empty:
         # attach names and return just the columns we want
         result = sorted_df.merge(
                business_df[['business_id','name']],
                on='business_id', how='left'
            )[['business_id','name','stars','categories']]
         return result, matched_user_id, sim

   # If no matches found among top_n users, return empty DataFrame
   return pd.DataFrame(columns=['business_id','name','stars','categories'])


if __name__ == "__main__":
   # ——— load your data from disk ———
   all_user_ratings_df = pd.read_pickle('data/user_ratings_gt_300.pkl')
   reviews_df          = pd.read_pickle('data/reviews.pkl')
   business_df         = pd.read_pickle('data/business.pkl')

   # loading a real user and processing it individually
   top_users_df = pd.read_pickle('data/user_ratings.pkl') # top 10 users with most reviews
   top_user_ids = top_users_df.index.tolist()
   target_user_id = top_user_ids[1]
   # ——— prompt for inputs ———
   target_user_id = input("Target user_id: ").strip()
   state          = input("State (e.g. CA): ").strip()
   city           = input("City (e.g. San Francisco): ").strip()
   cuisine        = input("Cuisine (e.g. Korean): ").strip()

   # run rec
   recs_df, matched_user_id, match_score = recommend_restaurants_for_user(
      target_user_id,
      all_user_ratings_df,
      reviews_df,
      business_df,
      state,
      city,
      cuisine,
      top_n=5
   )
   # ——— display results ———
   if recs_df.empty:
      print("No matching restaurants found.")
   else:
      print(f"Matched user {matched_user_id} ({match_score*100:.1f}% match)")
      print(recs_df.to_string(index=False))
