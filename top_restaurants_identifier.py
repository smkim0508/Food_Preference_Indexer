import pandas as pd

# finds the top_n most reviewed restaurants, default top 10
def find_most_reviewed_restaurants(business_df, reviews_df, top_n=10):
   # add review count
   review_counts = reviews_df.groupby('business_id').size().reset_index(name='review_count')
   
   # merge w/ business info
   business_info = business_df[['business_id', 'name', 'categories']]
   merged = pd.merge(business_info, review_counts, on='business_id', how='inner')
   
   # sort by descending review count
   most_reviewed = merged.sort_values(by='review_count', ascending=False)
   # return top_n restaurants
   return most_reviewed.head(top_n)

# for simple print test
if __name__ == "__main__":
   reviews_df = pd.read_pickle('data/reviews.pkl')
   business_df = pd.read_pickle('data/business.pkl')

   top_restaurants = find_most_reviewed_restaurants(
      business_df, 
      reviews_df, 
      top_n=10
   )

   print(top_restaurants)
   top_restaurants.to_csv('top_reviewed_restaurants.csv', index=False) # not necessary