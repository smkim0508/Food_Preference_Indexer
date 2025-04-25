import numpy as np
import os
from itertools import combinations

# scaled cosine similarity, accounting for magnitude
def scaled_cos_sim(a, b):
   norm_a = np.linalg.norm(a)
   norm_b = np.linalg.norm(b)
   cos_sim = np.dot(a, b) / (norm_a * norm_b) # calculate cosine similarity value
   mag_ratio = min(norm_a, norm_b) / max(norm_a, norm_b) # calculate ratio of magnitudes to scale
   return cos_sim * mag_ratio  # reduces score when magnitudes differ

# writing a script to take in .txt file and calculate cosine sim values for each pairing

# first load in data and parse through it into a list

folder_path = './' # set folder path as current directory
delimiter = '\n' # delimeter for files
ratings = [] # empty list

# loop and extract data from each file into ratings array
for filename in os.listdir(folder_path):
   if filename.endswith('.txt'): # could be changed to be more specific
      file_path = os.path.join(folder_path, filename)
      with open(file_path, 'r', encoding='utf-8') as file:
         content = file.read()
         parts = content.split(delimiter)  # split the file content using delimiter
         ratings.append(np.array(parts, dtype = 'float'))

# then use combinations function to make pairings
pairs = list(combinations(range(len(ratings)), 2))

# find the cos similarity values for each pair and save result
relative_index = {}
for i, j in pairs:
    sim = scaled_cos_sim(ratings[i], ratings[j])
    relative_index[(i, j)] = sim
    print(f"cosine similarity between vectors {i} and {j}: {sim:.4f}")