# multivariable regression rating predictor for ONE user
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# function to read designated cuisines
def load_cuisines(path='cuisines.txt'):
   with open(path, 'r', encoding='utf-8') as f:
      return [line.strip() for line in f if line.strip()]

# function to read valid US states
def load_states(path='states.txt'):
   with open(path, 'r', encoding='utf-8') as f:
      return [line.strip() for line in f if line.strip()]

# function to load in all the dataframes
def load_data(reviews_path, business_path, user_cuisine_path):
   reviews = pd.read_pickle(reviews_path)
   business = pd.read_pickle(business_path)
   user_cuisine = pd.read_pickle(user_cuisine_path)
   return reviews, business, user_cuisine

# function to process user data to find statistics about their reviews
def get_user_stats(reviews_df, user_id):
   """
   filters reviews for a single user and compute:
   - user_reviews: dataframe of just that user's reviews
   - user_mean: their average rating
   - user_harsh: difference from global mean (harshness/generosity), which can be positive or negative
   """
   global_mean = reviews_df['stars'].mean()
   user_reviews = reviews_df[reviews_df['user_id'] == user_id].copy()
   user_mean = user_reviews['stars'].mean()
   user_harsh = user_mean - global_mean
   return user_reviews, user_mean, user_harsh

# building featue matrix
def build_feature_matrix(user_reviews, business_df, user_cuisine, cuisines, user_mean, user_harsh):
   """
   build the feature matrix X and target vector y for one user's past review history.
   - user_reviews: their review rows
   - business_df: restaurant info
   - user_cuisine: all user's average ratings across each cuisine as a vector
   - cuisines: list of cuisines to refer to
   - user_mean, user_harsh: statistics specific to the target user that might sway values
   """
   # merge in business info
   df = user_reviews.merge(
      business_df[['business_id','stars','state','categories','latitude','longitude']],
      on='business_id', how='left'
   ).rename(columns={'stars_x':'stars_x','stars_y':'stars_y'})

   # add user cuisine trend features
   for c in cuisines:
      df[f'cuisine_{c}'] = user_cuisine.loc[df['user_id'].iloc[0], c]
   # add user mean & harshness
   df['user_mean'] = user_mean
   df['user_harsh'] = user_harsh

   # restaurant cuisine flags
   def flags_from_cuisines(cat_str):
      tags = {x.strip().lower() for x in (cat_str or '').split(',')}
      return {f'flag_{c}': int(c.lower() in tags) for c in cuisines}
   flags_df = df['categories'].apply(flags_from_cuisines).apply(pd.Series)
   df = pd.concat([df, flags_df], axis=1)
   print(flags_df)
   # flags_df.to_csv('flags.csv', index=False) # check output

   # states one-hot encoding
   states = load_states('states.txt')

   for s in states:
    df[f'state_{s}'] = (df['state'] == s).astype(int)

   # define feature columns
   feature_cols = (
      ['stars_y','latitude','longitude','user_mean','user_harsh']
      # + [f'cuisine_{c}' for c in cuisines]
      + [f'flag_{c}' for c in cuisines]
      + [f'state_{s}' for s in states]
   )
   print(feature_cols)

   df.to_csv('df_penultimate.csv', index=False)

   # build X and y, and fill any NaNs with 0
   X = df[feature_cols].fillna(0)
   X.to_csv('df_final.csv', index=False)

   y = df['stars_x']
   return X, y

# train the multivariable regressor and evaluate it
def train_and_evaluate(X, y, test_size=0.2, random_state=42, alpha=1.0):
   """
   split into train/validation, fit a Ridge regression pipeline,
   and return (fitted_model, rmse, predictions_on_val).
   """
   X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=test_size, random_state=random_state
   )
   pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('ridge', Ridge(alpha=alpha))
   ])
   pipeline.fit(X_train, y_train)
   preds = pipeline.predict(X_val)
   rmse = np.sqrt(mean_squared_error(y_val, preds))
   return pipeline, rmse, preds

def main():
   # load everything
   cuisines = load_cuisines('cuisines.txt')
   reviews, business, user_cuisine = load_data(
      'data/reviews.pkl',
      'data/business.pkl',
      'data/user_ratings_gt_300.pkl'
   )

   # pick one user
   top_users = pd.read_pickle('data/user_ratings.pkl').index.tolist()
   uid = top_users[1]

   # build features & target
   user_reviews, user_mean, user_harsh = get_user_stats(reviews, uid)
   X, y = build_feature_matrix(user_reviews, business, user_cuisine, cuisines, user_mean, user_harsh)

   # train & eval
   model, rmse, preds = train_and_evaluate(X, y)

   print(f"Trained Ridge regression for user {uid}")
   print(f"Validation RMSE: {rmse:.4f}")
   print("Sample predictions:", preds[:])

if __name__ == "__main__":
   main()
