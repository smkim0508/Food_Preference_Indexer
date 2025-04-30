import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# load data
reviews_df = pd.read_pickle('data/reviews.pkl')
business_df = pd.read_pickle('data/business.pkl')

top_users_df = pd.read_pickle('data/user_ratings.pkl') # top 10 users with most reviews
top_user_ids = top_users_df.index.tolist()

# pick specific user
target_user_id = top_user_ids[0]

# get all restaurants the user reviewed
user_reviews = reviews_df[reviews_df['user_id'] == target_user_id]

# merge with business features
user_reviews = user_reviews.merge(business_df, on='business_id', how='left')

# select numeric features
basic_features = ['stars', 'review_count', 'state', 'city']

# df_cuisines
# load in cuisines.txt to parse data
cuisines = []
delimiter = '\n'

# use short cuisines list for testing
with open('cuisines.txt', 'r', encoding='utf-8') as file:
   cuisines = file.read().split(delimiter)

for cuisine in cuisines:
   # multi-hot encode


print(categories_encoded)

# combine features
basic_features_array = user_reviews[basic_features].fillna(0).values
X_train_full = np.hstack([basic_features_array, categories_encoded])

# create y_train
y_train = user_reviews['stars_x'].values

# convert to torch tensors
X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

print(f"X_train_tensor shape: {X_train_tensor.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")

# define the model
class UserPreferenceModel(nn.Module):
    def __init__(self, input_dim):
        super(UserPreferenceModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train_tensor.shape[1]  # Correct input dimension now based on real data
model = UserPreferenceModel(input_dim)

# implementation details
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

# train
for epoch in range(epochs):
    model.train()
    
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_train_tensor)

# show top few results
for true, pred in zip(y_train_tensor[:10], predictions[:10]):
    print(f"True: {true.item():.2f}, Predicted: {pred.item():.2f}")
