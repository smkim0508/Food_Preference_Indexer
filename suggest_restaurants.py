import pandas as pd
from find_most_similar_user import find_top_matches

def get_target_user_reviews(target_user_id, reviews_df):
    return reviews_df[reviews_df['user_id'] == target_user_id]

def get_user_reviewed_restaurants(user_id, reviews_df):
    return reviews_df[reviews_df['user_id'] == user_id]

def filter_by_city(restaurants_df, target_city, business_df):
    # merge to get city info
    restaurants_df = restaurants_df.merge(business_df[['business_id', 'city']], on='business_id', how='left')
    return restaurants_df[restaurants_df['city'] == target_city]

def filter_by_cuisine(restaurants_df, target_cuisine, business_df):
    # merge to get categories info
    restaurants_df = restaurants_df.merge(business_df[['business_id', 'categories']], on='business_id', how='left')
    # only restaurants containing the target cuisine in their categories
    return restaurants_df[restaurants_df['categories'].fillna('').str.contains(target_cuisine, case=False)]

def sort_by_high_rating(restaurants_df):
    return restaurants_df.sort_values(by='stars', ascending=False)

# load in all users data
# all_user_ratings_df = pd.read_pickle('user_ratings.pkl') # to load top 10 users with most reviews
all_user_ratings_df = pd.read_pickle('user_ratings_gt_300.pkl') # to load all users with more than 300 reviews
reviews_df = pd.read_pickle('reviews.pkl')
business_df = pd.read_pickle('business.pkl')
# load in the target user to compare with
# target_user_ratings_df = pd.read_pickle(target_user_path)
# top_users_df = pd.read_pickle('user_ratings.pkl') # top 10 users with most reviews
# top_user_ids = top_users_df.index.tolist()

# target user
# target_user_id = top_user_ids[0] # change this to take user input later

target_user_ratings_df = pd.read_pickle('dummy_ratings.pkl')

target_city = 'San Francisco'
target_cuisine = 'Korean'

# get target user's past reviews
# target_user_reviews = get_target_user_reviews(target_user_id, reviews_df)

# find top similar users
top_matches = find_top_matches(all_user_ratings_df, target_user_ratings_df, top_n=5)

# print(top_matches)

# go through top matches
for idx, (user_idx, matched_user_id, sim) in enumerate(top_matches):
    print(f"Trying recommendations from similar user: {matched_user_id} (similarity {sim:.4f})")
    
    # find restaurants this user reviewed
    user_restaurants = get_user_reviewed_restaurants(matched_user_id, reviews_df)
    
    # filter by city
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
