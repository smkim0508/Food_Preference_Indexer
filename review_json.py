import json
import pandas as pd

# load the Yelp Dataset in JSON file
data = []
with open("dataset/yelp_academic_dataset_reviews.json", "r", encoding="utf-8") as data_file:
    for line in data_file:
        data.append(json.loads(line))

# turn into pandas df
df = pd.DataFrame(data)

# save as a .pkl file
df.to_pickle('data/reviews.pkl')