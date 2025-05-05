import pandas as pd

# load in data and print num rows
df = pd.read_pickle('data/users.pkl')

num_rows = df.shape[0]
print(f"The df has {num_rows} rows")
