import pandas as pd
import random

# load in cuisines data
cuisines = []
delimiter = '\n'

with open('cuisines.txt', 'r', encoding='utf-8') as file:
   cuisines = file.read().split(delimiter)

# generate sample rating for each cuisine
user_ratings = []
user2_ratings = []
user3_ratings = []

for i, cuisine in enumerate(cuisines):
   user_ratings.append(5) # a dummy user with all 5 star ratings
   user2_ratings.append(1) # a dummy user with all 1 star ratings
   user3_ratings.append(5*random.random()) # a dummy user with random ratings in range [0, 5]

# checking
ratings_df = pd.DataFrame({
    'cuisine': cuisines,
    'user1': user_ratings,
    'user2': user2_ratings,
    'user3': user3_ratings
})

print(ratings_df)