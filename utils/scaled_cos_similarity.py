import numpy as np

# scaled cosine similarity, accounting for magnitude
def scaled_cos_sim(a, b):
   norm_a = np.linalg.norm(a)
   norm_b = np.linalg.norm(b)

   # edge case to handle 0 norm
   if ((norm_a == 0) or (norm_b == 0)): return -1

   cos_sim = np.dot(a, b) / (norm_a * norm_b) # calculate cosine similarity value
   mag_ratio = min(norm_a, norm_b) / max(norm_a, norm_b) # calculate ratio of magnitudes to scale
   return cos_sim * mag_ratio  # reduces score when magnitudes differ