from collections import Counter
import numpy as np
import math

# Your documents list
documents = [
    ["bird", "cat", "cat", "dog", "dog", "bird", "tiger", "tiger"], #1
    ["cat", "tiger", "cat", "dog"], #2
    ["dog", "bird", "bird", "cat"], #3
    ["cat", "tiger", "cat", "dog"], #4
    ["tiger", "tiger", "dog", "tiger"], #5
    ["cat", "cat", "tiger", "tiger"], #6
    ["bird", "cat", "bird"], #7
    ["bird", "cat", "dog"], #8
    ["dog", "cat", "tiger"], #9
    ["tiger", "tiger", "tiger"], #10
    ["bird", "cat", "cat", "tiger"], #11
    ["bird", "cat", "tiger", "tiger", "dog"], #12
    ["bird", "cat", "tiger", "dog"], #13
    ["bird", "cat", "tiger"], #14
    ["bird", "bird", "bird"], #15
]

query = ["bird", "bird", "bird", "tiger"]
keywords = ["bird", "cat", "dog", "tiger"]

# Initialize document keyword counts as a 2D array
doc_counts = np.zeros((len(keywords), len(documents)), dtype=int)

# Fill the document counts array
for i, document in enumerate(documents):
    counts = Counter(document)
    for j, keyword in enumerate(keywords):
        doc_counts[j, i] = counts.get(keyword, 0)

query_counts_array = np.zeros((len(keywords), 1), dtype=int)
query_counts = Counter(query)
for j, keyword in enumerate(keywords):
    query_counts_array[j, 0] = query_counts.get(keyword, 0)

# Perform SVD
U, S, VT = np.linalg.svd(doc_counts, full_matrices=False)
S_matrix = np.diag(S)

# Print the results
print("Document Counts:\n", doc_counts)
print("\nU matrix:\n", U)
print("\nSingular values (S):\n", S_matrix)
print("\nVT matrix:\n", VT)
print("\nV matrix:\n", VT.T)

# Rank 2 Approximation
U_rank_2 = U[:, :2]  # First two columns of U
S_rank_2 = np.diag(S_matrix[:2])  # Diagonal matrix of the first two singular values
VT_rank_2 = VT[:2, :]  # First two rows of VT
S_rank_2_Matrix = np.diag(S_rank_2)
print("\nU matrix:\n", U_rank_2)
print("\nSingular values (S):\n", S_rank_2_Matrix)
print("\nVT matrix:\n", VT_rank_2)
print("\nV matrix:\n", VT_rank_2.T)

Topic_doc = S_rank_2_Matrix @ VT_rank_2
# Calculate the Rank 2 Approximation
print("\nTopic_doc:\n", Topic_doc.T)
Topic_doc_reshaped = Topic_doc.T.reshape(-1, 2)  # Ensure it reshapes to (15, 2)

# Print the reshaped Topic_doc
print("\nReshaped Topic_doc (15 documents, 2 topics):\n", Topic_doc_reshaped)

S_rank_2_Matrix_inv = np.zeros_like(S_rank_2_Matrix)  # Initialize the inverse matrix
for i in range(S_rank_2_Matrix.shape[0]):
    if S_rank_2_Matrix[i, i] != 0:  # Avoid division by zero
        S_rank_2_Matrix_inv[i, i] = 1 / S_rank_2_Matrix[i, i]  # Take the reciprocal

print("\nInverse of Reduced S:\n", S_rank_2_Matrix_inv)

# New query computation
New_query = (query_counts_array.T @ U_rank_2 @ S_rank_2_Matrix_inv).flatten()  # Flatten to ensure it's 1D
print("\nNew_query:\n", New_query)

# Calculate similarity
sim = []
for i in range(len(documents)):
    similarity = (New_query[0] * Topic_doc_reshaped[i][0] + New_query[1] * Topic_doc_reshaped[i][1]) / \
                 (math.sqrt(New_query[0]**2 + New_query[1]**2) * math.sqrt(Topic_doc_reshaped[i][0]**2 + Topic_doc_reshaped[i][1]**2))
    sim.append([f"doc{i + 1}", similarity])

# Print the similarity results
print("\nSimilarities:\n", sim)
sorted_sim = sorted(sim, key=lambda x: x[1], reverse=True)

# Print the ranked results
print("\nRanked Similarities (from max to min):")
for rank, (doc, score) in enumerate(sorted_sim, start=1):
    print(f"Rank {rank}: {doc} with similarity score {score:.8f}")