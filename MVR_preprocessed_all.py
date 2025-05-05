# Multivariable Regression with preprocessed data, for quick testing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm 

# load data
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

def train_and_evaluate(X, y, test_size=0.2, random_state=42, alpha=1.0):
   X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=test_size, random_state=random_state
   )
   pipeline = Pipeline([
      ('scaler', StandardScaler()),
      # ('ridge',  Ridge(alpha=10))
      ('lasso', Lasso(alpha=10))
   ])
   pipeline.fit(X_train, y_train)
   preds = pipeline.predict(X_val)
   preds_rounded = np.rint(preds)

   # build df for test
   df_preds = pd.DataFrame()
   df_preds['y_true']       = y_val.reset_index(drop=True)
   df_preds['y_pred_raw']   = preds
   df_preds['y_pred_round'] = preds_rounded

   # save to CSV
   df_preds.to_csv('predictions.csv')

   # feature names
   try:
      feature_names = X.columns.tolist()
   except AttributeError:
      feature_names = [f"f{i}" for i in range(X.shape[1])]

   # take lasso or ridge
   if 'lasso' in pipeline.named_steps:
      model = pipeline.named_steps['lasso']
   else:
      model = pipeline.named_steps['ridge']

   # make feature coefficient pairs
   coefs = model.coef_
   feat_imp = list(zip(feature_names, coefs))

   # prune sparse values for lasso
   if hasattr(model, 'alpha') and isinstance(model, __import__('sklearn').linear_model.Lasso):
      feat_imp = [(f, w) for f, w in feat_imp if w != 0]

   # sort by importance
   feat_imp = sorted(feat_imp, key=lambda x: abs(x[1]), reverse=True)

   # print the top n things; n = 20 for testing
   top_n = 20
   print(f"\nTop {top_n} features:")
   for f, w in feat_imp[:top_n]:
      print(f"  {f:20s}  {w: .4f}")
   
   # graphing the data...
   # scatter plot: True vs Predicted (Rounded)
   # plt.figure(figsize=(8, 6))
   # plt.scatter(df_preds['y_true'], df_preds['y_pred_raw'], alpha=0.6)
   # plt.plot([df_preds['y_true'].min(), df_preds['y_true'].max()],
   #          [df_preds['y_true'].min(), df_preds['y_true'].max()],
   #          'r--', label='Perfect Prediction')
   # plt.xlabel('True Ratings')
   # plt.ylabel('Predicted Ratings (Rounded)')
   # plt.title('Ridge Regression Predictions vs True Ratings')
   # plt.legend()
   # plt.grid(True)
   # plt.tight_layout()

   # heat/density map
   plt.figure(figsize=(8, 6))
   plt.hexbin(df_preds['y_true'], df_preds['y_pred_round'], gridsize=50, cmap='viridis', bins='log')
   plt.plot([df_preds['y_true'].min(), df_preds['y_true'].max()],
            [df_preds['y_true'].min(), df_preds['y_true'].max()],
            'r--', label='Perfect Prediction')
   plt.colorbar(label='log10(N)')
   plt.xlabel('True Values')
   plt.ylabel('Predicted Values')
   plt.title('Prediction Density: True vs Predicted')
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   # plt.show()
   # plt.savefig('MVR_predictions_1st.png')
   plt.savefig('MVR_lasso_predictions_rounded_heatmap_02.png')
   plt.close()  # optional but safe in loops/scripts

   rmse  = np.sqrt(mean_squared_error(y_val, preds_rounded))
   # rmse  = np.sqrt(mean_squared_error(y_val, preds))
   # print(preds_rounded)
   print(preds)
   return pipeline, rmse

# testing
def main():
   cuisines, states = load_cuisines('cuisines.txt'), load_states('states.txt')
   reviews, business, user_cuisine = load_data(
      'data/reviews.pkl',
      'data/business.pkl',
      'data/user_ratings_gt_300.pkl'
   )

   X = pd.read_csv('features.csv')
   y = pd.read_csv('target.csv')
   
   model, rmse = train_and_evaluate(X, y)

   print(f"Trained on {X.shape[0]} examples with {X.shape[1]} features")
   print(f"Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
   main()
