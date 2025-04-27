import numpy as np
import os
from itertools import combinations
from scaled_cos_similarity import scaled_cos_sim

# writing a script to take in .txt file and calculate cosine sim values for each pairing

# first load in data and parse through it into a list

folder_path = './' # set folder path as current directory
delimiter = '\n' # delimeter for files
ratings = [] # empty list

# loop and extract data from each file into ratings array
for filename in os.listdir(folder_path):
   if filename.startswith('test') and filename.endswith('.txt'): # could be changed to be more specific
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