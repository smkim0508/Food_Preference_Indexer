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
# print(merged_df['categories'].unique())

# asian = sorted_df[sorted_df['categories'].str.contains('Asian', na=False)]
# print(asian.head(10))

# chinese = sorted_df[sorted_df['categories'].str.contains('Chinese', na=False)]
# print(chinese.head(10))

# korean = sorted_df[sorted_df['categories'].str.contains('Korean', na=False)]
# print(korean.head(10))

# load in cuisines.txt to parse data
df_cuisines = {}
cuisines = []
delimiter = '\n'

with open('cuisines.txt', 'r', encoding='utf-8') as file:
   content = file.read()
   # split the file content using delimiter
   cuisine = content.split(delimiter)
   cuisines.append(cuisine)

# creating a dict where each keyword = cuisine, each value = pandas df that contains sorted_df filtered by each cuisine category
for cuisine in cuisines[0]: #flatten cuisines list
    keyword = cuisine
    # create separate df for each cuisine in cuisines list
    df_cuisines[keyword] = sorted_df[sorted_df['categories'].str.contains(keyword, case=False, na=False)]

# test with one of the cuisines
print(df_cuisines['Taiwanese'])

