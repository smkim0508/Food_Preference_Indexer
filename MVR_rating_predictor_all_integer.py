import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def load_cuisines(path='cuisines.txt'):
   with open(path,'r',encoding='utf-8') as f:
      return [line.strip() for line in f if line.strip()]

def load_states(path='states.txt'):
   with open(path,'r',encoding='utf-8') as f:
      return [line.strip() for line in f if line.strip()]

def load_data(reviews_path, business_path, user_cuisine_path):
   reviews = pd.read_pickle(reviews_path)
   business = pd.read_pickle(business_path)
   user_cuisine = pd.read_pickle(user_cuisine_path)
   return reviews, business, user_cuisine

def build_feature_matrix_all(reviews, business, user_cuisine, cuisines, states):
   """
   Returns:
   X: DataFrame of features for every (user, restaurant) review
   y: Series of that user’s actual star rating for that row
   """

   # for each user
   global_mean = reviews['stars'].mean()
   user_stats = reviews.groupby('user_id')['stars']\
                     .agg(user_mean='mean')\
                     .assign(user_harsh=lambda df: df['user_mean'] - global_mean)
   user_stats['user_review_count'] = reviews.groupby('user_id').size()
   print('checkpoint 1')

   # merge reviews with user_stats
   df = reviews.merge(user_stats, left_on='user_id', right_index=True, how='left')
   print('checkpoint 2')

   # merge in user_cuisine averages
   cuisine_cols = user_cuisine.columns.tolist()
   cmap = {c: f'cuisine_{c}' for c in cuisine_cols} # each column named cuisines_{name of cuisine}, e.g. cuisines_asian
   uc_renamed = user_cuisine.rename(columns=cmap)
   df = df.merge(uc_renamed, left_on='user_id', right_index=True, how='left')
   print('checkpoint 3')

   # merge in restaurant info
   df = df.merge(
      business[['business_id','stars','state','categories','latitude','longitude']],
      on='business_id', how='left', suffixes=('_user','_yelp')
   )
   print('checkpoint 4')

   # one-hot-encode flags for cuisines
   def flags_from_cuisines(cat_str):
      tags = {x.strip().lower() for x in (cat_str or '').split(',')}
      return {f'flag_{c}': int(c.lower() in tags) for c in cuisines}

   flags = df['categories'].apply(flags_from_cuisines).apply(pd.Series)
   df = pd.concat([df, flags], axis=1)
   print('checkpoint 5')

   # one-hot-encode flags for states
   for s in states:
      df[f'state_{s}'] = (df['state'] == s).astype(int)
   print('checkpoint 6')

   # compile all features
   feature_cols = (
      ['stars_yelp','latitude','longitude','user_mean','user_harsh', 'user_review_count']
      + [f'cuisine_{c}' for c in cuisines]
      + [f'flag_{c}'    for c in cuisines]
      + [f'state_{s}'    for s in states]
   )
   # find feature and target
   X = df[feature_cols].fillna(0)
   y = df['stars_user']

   X.to_csv('features.csv', index=False) # saves all feature columns
   y.to_frame(name='stars_user').to_csv('target.csv', index=False) # save y as its own CSV
   return X, y

def train_and_evaluate(X, y, test_size=0.2, random_state=42, alpha=1.0):
   X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=test_size, random_state=random_state
   )
   print('checkpoint 7')
   pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('ridge',  Ridge(alpha=alpha))
   ])
   pipeline.fit(X_train, y_train)
   preds = pipeline.predict(X_val)
   preds_rounded = np.rint(preds)

   # testing df
   df_preds = pd.DataFrame()
   df_preds['y_true']       = y_val.reset_index(drop=True)
   df_preds['y_pred_raw']   = preds
   df_preds['y_pred_round'] = preds_rounded
   df_preds.to_csv('predictions.csv')

   rmse  = np.sqrt(mean_squared_error(y_val, preds_rounded))
   print(preds_rounded)
   return pipeline, rmse

# testing below
def main():
   cuisines, states = load_cuisines('cuisines.txt'), load_states('states.txt')
   reviews, business, user_cuisine = load_data(
      'data/reviews.pkl',
      'data/business.pkl',
      'data/user_ratings_gt_300.pkl'
   )

   X, y = build_feature_matrix_all(reviews, business, user_cuisine, cuisines, states)
   model, rmse = train_and_evaluate(X, y)

   print(f"Trained on {X.shape[0]} examples with {X.shape[1]} features")
   print(f"Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
   main()
