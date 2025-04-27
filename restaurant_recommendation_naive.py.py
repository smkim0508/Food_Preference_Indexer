import pandas as pd

# load data
reviews_df = pd.read_pickle('reviews.pkl')
business_df = pd.read_pickle('business.pkl')

# specify user and target restaurant
target_user_id = 'bYENop4BuQepBjM1-BI3fA' # dummy ex
target_business_id = 'c8GjPIOTGVmIemT7j5_SyQ' # dummy ex

# get target restaurant's categories
target_categories = business_df.loc[business_df['business_id'] == target_business_id, 'categories'].values
if len(target_categories) == 0:
   raise ValueError("Business ID not found.")
target_categories = target_categories[0].split(', ')

# find the restaurants reviewed by target user
target_reviews = reviews_df[reviews_df['user_id'] == target_user_id]
target_reviews = target_reviews.merge(business_df[['business_id', 'categories']], on='business_id', how='left')

# def overlap function to count overlaps of categories, used to weight
def count_category_overlap(row):
   if pd.isna(row['categories']):
      return 0
   business_categories = row['categories'].split(', ')
   overlap = len(set(target_categories) & set(business_categories))
   return overlap

# calculate overlaps and filter restaurants with at least 1 reviews by target user
target_reviews['overlap_count'] = target_reviews.apply(count_category_overlap, axis=1)
matching_reviews = target_reviews[target_reviews['overlap_count'] > 0]

# calculate weighted average using overlap count
if not matching_reviews.empty:
   weighted_sum = (matching_reviews['stars'] * matching_reviews['overlap_count']).sum()
   total_weight = matching_reviews['overlap_count'].sum()
   weighted_avg_score = weighted_sum / total_weight
else:
   weighted_avg_score = None 

print(f"Predicted weighted liking score for {target_business_id}: {weighted_avg_score}")

# parse the target restaurant's rating
yelp_rating = business_df[business_df['business_id'] == target_business_id]['stars'].values[0]
print(yelp_rating)

# calculate the recommended score based on following equation:
# [((0.1/(1.1 - (my rating/5)))-0.55) * (0.1/(business rating/5))] + business rating
# gives advantage to my rating being really high and penalizes if bad, also scales less if the business rating is already high

final_score = min(((0.1/(1.1-(weighted_avg_score/5)))-0.55) * (0.1/(yelp_rating/5)) + yelp_rating, 5) # min() ensures max score of 5

print(final_score)