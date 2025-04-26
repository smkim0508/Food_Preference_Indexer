import pandas as pd

#load in df from .pkl file
df = pd.read_pickle('reviews.pkl')

# 1. Group by 'business_id' and count the number of unique 'user_id's
business_user_counts = df.groupby('business_id')['user_id'].nunique().reset_index()

# 2. Rename the column for clarity
business_user_counts = business_user_counts.rename(columns={'user_id': 'unique_user_count'})

# 3. Sort by unique user count descending
top_businesses = business_user_counts.sort_values(by='unique_user_count', ascending=False)

# 4. Select the top 100
top_100_businesses = top_businesses.head(3)

# # 5. Display
# print(top_100_businesses)


import pandas as pd

# Assuming:
# - df has 'user_id' and 'business_id'
# - top_100_businesses is a DataFrame with 'business_id' column

# Step 1: Filter reviews to only those related to top 100 businesses
filtered_reviews = df[df['business_id'].isin(top_100_businesses['business_id'])]

# Step 2: Group by user_id, and collect which businesses they reviewed
user_businesses = filtered_reviews.groupby('user_id')['business_id'].apply(set)

# Step 3: Identify users who reviewed *all* top 100 businesses
top_100_set = set(top_100_businesses['business_id'])
qualified_users = user_businesses[user_businesses.apply(lambda reviewed: top_100_set.issubset(reviewed))]

# Step 4: View the result
print(qualified_users.index.tolist())  # List of user_ids who reviewed all top 100 businesses
