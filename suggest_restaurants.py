import pandas as pd
from find_most_similar_user import find_top_matches

# returns the restaurants that was reviewed by a user, given user_id
def get_user_reviewed_restaurants(user_id, reviews_df):
   return reviews_df[reviews_df['user_id'] == user_id]

# filter a list of restaurants by state
def filter_by_state(restaurants_df, target_state, business_df):
   restaurants_df = restaurants_df.merge(business_df[['business_id', 'state']], on='business_id', how='left')
   return restaurants_df[restaurants_df['state'] == target_state]

# filter a list of restaurants by city
def filter_by_city(restaurants_df, target_city, business_df):
   # merge to get city info
   restaurants_df = restaurants_df.merge(business_df[['business_id', 'city']], on='business_id', how='left')
   return restaurants_df[restaurants_df['city'] == target_city]

# filter a list of restaurants by cuisine
def filter_by_cuisine(restaurants_df, target_cuisine, business_df):
   # merge to get categories info
   restaurants_df = restaurants_df.merge(business_df[['business_id', 'categories']], on='business_id', how='left')
   # only restaurants containing the target cuisine in their categories
   return restaurants_df[restaurants_df['categories'].fillna('').str.contains(target_cuisine, case=False)]

# sort a list of restaurants by highest ratings
def sort_by_high_rating(restaurants_df):
   return restaurants_df.sort_values(by='stars', ascending=False)

# load in all users data
all_user_ratings_df = pd.read_pickle('user_ratings_gt_300.pkl') # to load all users with more than 300 reviews
# load in raw reviews and business info
reviews_df = pd.read_pickle('reviews.pkl')
business_df = pd.read_pickle('business.pkl')

# using preprocess_individual, come up with a way to parse each user rating given their user_id
target_user_ratings_df = pd.read_pickle('dummy_ratings.pkl') # dummy for now

# set targets, also should be loaded in somehow
# dummy values for now
target_city = 'San Francisco' 
target_cuisine = 'Korean'
target_state = 'CA'

# find top similar users
top_matches = find_top_matches(all_user_ratings_df, target_user_ratings_df, top_n=5)

# go through top matches
for idx, (user_idx, matched_user_id, sim) in enumerate(top_matches):
   print(f"Trying recommendations from similar user: {matched_user_id} (similarity {sim:.4f})")
   
   # find restaurants this user reviewed
   user_restaurants = get_user_reviewed_restaurants(matched_user_id, reviews_df)
   
   # filter by state, doesn't work well with small user data
   # state_filtered = filter_by_state(user_restaurants, target_cuisine, business_df)

   # filter by city, doesn't work well with small user data
#  city_filtered = filter_by_city(user_restaurants, target_city, business_df)
   
   # filter by cuisine
#  cuisine_filtered = filter_by_cuisine(city_filtered, target_cuisine, business_df)
   cuisine_filtered = filter_by_cuisine(user_restaurants, target_cuisine, business_df)
   
   # sort by rating
   final_recommendations = sort_by_high_rating(cuisine_filtered)

   if not final_recommendations.empty:
      print(f"Found {len(final_recommendations)} matching restaurants!")
      print(final_recommendations[['business_id', 'stars', 'categories']].head(10))
      break  # stop once you find matching restaurants
   else:
      print(f"No matches for user {matched_user_id}, trying next...")
