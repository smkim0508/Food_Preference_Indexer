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

   X = sm.add_constant(X)
   result = sm.OLS(y, X).fit()
   print(result.summary())

   return result

def main():
   cuisines, states = load_cuisines('cuisines.txt'), load_states('states.txt')
   reviews, business, user_cuisine = load_data(
      'data/reviews.pkl',
      'data/business.pkl',
      'data/user_ratings_gt_300.pkl'
   )

   X = pd.read_csv('features.csv')
   y = pd.read_csv('target.csv')
   
   model  = train_and_evaluate(X, y)

   # print(f"Trained on {X.shape[0]} examples with {X.shape[1]} features")
   # print(f"Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
   main()
