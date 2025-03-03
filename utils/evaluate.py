import numpy as np

def recall_at_n(query_features, reference_features, query_labels, reference_labels, n=1):
    num_queries = query_features.shape[0]
    recall_count = 0

    # Iterate over each query
    for i in range(num_queries):
        query_feature = query_features[i]
        query_label = query_labels[i]

        # Compute distance
        distances = np.linalg.norm(reference_features - query_feature, axis=1)
        
        # Sort the indices by top N correlated candidates
        top_n_indices = np.argsort(distances)[:n]

        # Check if query is in the top N retrieved candidates
        if query_label in reference_labels[top_n_indices]:
            recall_count += 1
    recall_at_n_score = recall_count / num_queries
    return recall_at_n_score

# Example data
query_features = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # Shape: (3, 2)
reference_features = np.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1], [4.0, 5.0]])  # Shape: (4, 2)
query_labels = np.array([0, 1, 2])  # Labels for query images
reference_labels = np.array([0, 1, 2, 3])  # Labels for reference images

# Compute Recall@1
recall_at_1 = recall_at_n(query_features, reference_features, query_labels, reference_labels, n=1)
print("Recall@1:", recall_at_1)

# Compute Recall@2
recall_at_2 = recall_at_n(query_features, reference_features, query_labels, reference_labels, n=2)
print("Recall@2:", recall_at_2)