import json
import pandas as pd

# load the Yelp Business Dataset in JSON file
data = []
with open("yelp_academic_dataset_business.json", "r", encoding="utf-8") as data_file:
    for line in data_file:
        data.append(json.loads(line))

# turn into pandas df
df = pd.DataFrame(data)

# save as a .pkl file
df.to_pickle('business.pkl')