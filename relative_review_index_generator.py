import numpy as np

# test
A = np.array([5, 5, 5])
B = np.array([5, 5, 1])

# compute cosine similarity
# cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# print("Cosine Similarity:", cos_sim)

# naive testing w/ pure cosine, need to account for magnitude too since we want diff reviews to be harshly viewed

# testing with scaled cosine similarity, accounting for magnitude
def scaled_cosine_penalty(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    mag_ratio = min(np.linalg.norm(a), np.linalg.norm(b)) / max(np.linalg.norm(a), np.linalg.norm(b))
    print(mag_ratio)
    return cos_sim * mag_ratio  # reduces score when magnitudes differ

print(scaled_cosine_penalty([5, 5, 1], [5, 5, 5])) 

# need to verify math
