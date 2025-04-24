import numpy as np

# test
A = np.array([5, 5, 5, 5, 5])
B = np.array([1, 0, 0, 5, 1])

# compute cosine similarity
cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

print("Cosine Similarity:", cos_sim)

# naive testing w/ pure cosine, need to account for magnitude too since we want diff reviews to be harshly viewed

