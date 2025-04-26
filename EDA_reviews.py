# performing EDA with reviews dataset to efficiently group

import pandas as pd

#load in df from .pkl file
df = pd.read_pickle('reviews.pkl')
business_df = pd.read_pickle('business.pkl')

# check the column names
# print(df.columns.tolist())

# merge dfs using matching business_id
merged_df = pd.merge(df, business_df[['business_id', 'categories']], on='business_id', how='left')

sorted_df = merged_df.sort_values(by=['business_id', 'categories'])

# print(sorted_df.head(100))
# check all of the different categories
print(merged_df['categories'].unique())