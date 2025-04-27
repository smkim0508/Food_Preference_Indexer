import pandas as pd

#load in df from .pkl file
df = pd.read_pickle('reviews.pkl')

business_user_counts = df.groupby('business_id')['user_id'].nunique().reset_index()
business_user_counts = business_user_counts.rename(columns={'user_id': 'unique_user_count'})

top_businesses = business_user_counts.sort_values(by='unique_user_count', ascending=False)

top_100_businesses = top_businesses.head(100)

# Assuming:
# - df has 'user_id' and 'business_id'
# - top_100_businesses is a DataFrame with 'business_id' column

# filter reviews only related to top 100 businesses
filtered_reviews = df[df['business_id'].isin(top_100_businesses['business_id'])]

# group by user_id, collect which businesses they reviewed
user_businesses = filtered_reviews.groupby('user_id')['business_id'].apply(set)

# identify users who reviewed all top 100 businesses
top_100_set = set(top_100_businesses['business_id'])
qualified_users = user_businesses[user_businesses.apply(lambda reviewed: top_100_set.issubset(reviewed))]

print(qualified_users.index.tolist())
# determined this is not feasible in real-world data. resort to a different method.
