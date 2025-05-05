import pandas as pd

# check the average ratings across all reviews

reviews = pd.read_pickle('data/reviews.pkl')
avg_rating = reviews['stars'].mean()

print(f"Average rating across all {len(reviews)} reviews: {avg_rating:.2f}")
