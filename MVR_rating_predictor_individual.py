# multivariable regression rating predictor for ONE user
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# load the cuisines list from cuisines.txt
with open('cuisines.txt', 'r', encoding='utf-8') as f:
    cuisines = [line.strip() for line in f if line.strip()]

# load data
reviews      = pd.read_pickle('data/reviews.pkl')
business     = pd.read_pickle('data/business.pkl')
user_cuisine = pd.read_pickle('data/user_ratings_gt_300.pkl')  # user × cuisine avg

# dummy user / business
top_business_df    = pd.read_csv('top_reviewed_restaurants.csv')
business_ids       = top_business_df[top_business_df.columns[0]].tolist()
top_users_df       = pd.read_pickle('data/user_ratings.pkl')
target_business_id = business_ids[0]

# parse global user stats
global_mean  = reviews['stars'].mean()
uid     = top_users_df.index.tolist()[1]
user_reviews = reviews[reviews.user_id == uid].copy()
user_mean    = user_reviews['stars'].mean()
user_harsh   = user_mean - global_mean

# merge with business info
df = user_reviews.merge(
    business[['business_id','stars','state','categories','latitude','longitude']],
    on='business_id', how='left'
).rename(columns={'stars_x':'stars_x','stars_y':'stars_y'})

# USER FEATURES (same for every row)
#    a) cuisine-trend vector (one col per cuisine)
for cuisine in cuisines:
    df[f'cuisine_{cuisine}'] = user_cuisine.loc[uid, cuisine]

#    b) overall mean & harshness
df['user_mean']  = user_mean
df['user_harsh'] = user_harsh

# RESTAURANT FEATURES
#     a) cuisine flags (1 if restaurant’s categories include it)
def flags_from_cuisines(cat_str):
    tags = {c.strip().lower() for c in (cat_str or '').split(',')}
    return {f'flag_{c}': int(c.lower() in tags) for c in cuisines}

flags_df = df['categories'].apply(flags_from_cuisines).apply(pd.Series)
df = pd.concat([df, flags_df], axis=1)

#     b) state one-hot encoding
state_dummies = pd.get_dummies(df['state'], prefix='state')
df = pd.concat([df, state_dummies], axis=1)

# build features and target
feature_cols = (
    ['stars_y','latitude','longitude','user_mean','user_harsh']
    + [f'cuisine_{c}' for c in cuisines]
    + [f'flag_{c}'    for c in cuisines]
    + list(state_dummies.columns)
)
X = df[feature_cols]
y = df['stars_x']

# make sure no NaN values
X = X.fillna(0)

# train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# fit a simple Ridge regression pipeline
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
model.fit(X_train, y_train)

# eval
preds = model.predict(X_val)
print("RMSE:", np.sqrt(mean_squared_error(y_val, preds)))
print(preds)