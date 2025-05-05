import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


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

def train_and_evaluate_rf(X, y, test_size=0.2, random_state=42, n_estimators=100):
   X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=test_size, random_state=random_state
   )

   # building RF classifier
   model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
   model.fit(X_train, y_train)

   y_pred = model.predict(X_val)

   # save predictions
   df_preds = pd.DataFrame({
      'y_true': y_val,
      'y_pred': y_pred
   })
   df_preds.to_csv('rf_predictions.csv', index=False)

   # find accuracy
   acc = accuracy_score(y_val, y_pred)
   print(f"Accuracy: {acc:.4f}")
   print(classification_report(y_val, y_pred))

   # build confusion matrix
   cm = confusion_matrix(y_val, y_pred)
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4,5])
   disp.plot(cmap='Blues', values_format='d')
   plt.title("Confusion Matrix")
   
   plt.savefig('RF_predictions_1st.png')
   plt.close()

   # optional to grid search for best model
   # param_grid = {
   #    'n_estimators': [50,100,200],
   #    'max_depth':    [None,10,20],
   #    'min_samples_leaf': [1,2,4],
   #    'max_features': ['sqrt','log2']
   # }

   # grid = GridSearchCV(
   #    RandomForestClassifier(n_jobs=-1, random_state=42),
   #    param_grid,
   #    cv=5,
   #    scoring='accuracy',
   #    n_jobs=-1
   # )
   # grid.fit(X_train, y_train)
   # print("Best RF params:", grid.best_params_, "CV acc:", grid.best_score_)
   # print("Grid:", grid)
   # clf = grid.best_estimator_
   # print("best_estimator:", clf)

   return model, acc

# testing below
def main():
   cuisines, states = load_cuisines('cuisines.txt'), load_states('states.txt')
   reviews, business, user_cuisine = load_data(
      'data/reviews.pkl',
      'data/business.pkl',
      'data/user_ratings_gt_300.pkl'
   )

   X = pd.read_csv('features.csv')
   y = pd.read_csv('target.csv').values.ravel()
   
   model, acc = train_and_evaluate_rf(X, y)

   print(f"Trained on {X.shape[0]} examples with {X.shape[1]} features")
   print(f"Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
   main()
