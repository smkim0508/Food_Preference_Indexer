import pandas as pd

#load in df from .pkl file
df = pd.read_pickle('reviews.pkl')

# group by user_id
grouped = df.groupby('user_id')['business_id'].apply(list).reset_index()

# add a column for how many reviews each user wrote
grouped['review_count'] = grouped['business_id'].apply(len)

# filter users who wrote more than n reviews
n = int(input()) # user input
filtered = grouped[grouped['review_count'] > n]

# print all applicable rows
print(filtered)