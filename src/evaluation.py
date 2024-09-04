from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed

def precision_at_k(relevance, k):
    return np.sum(relevance[:k]) / k

def recall_at_k(relevance, all_relevant):
    return np.sum(relevance[:len(relevance)]) / all_relevant

def mean_reciprocal_rank(relevance):
    for i, rel in enumerate(relevance):
        if rel == 1:
            return 1 / (i + 1)
    return 0

# Function to compute NDCG@K
def ndcg_at_k(relevance, k):
    relevance = np.array(relevance)  # Ensure relevance is a NumPy array
    if k > len(relevance):
        k = len(relevance)

    # Calculate DCG@K
    dcg = np.sum((2 ** relevance[:k] - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate ideal DCG@K (best possible order of relevance)
    ideal_relevance = sorted(relevance, reverse=True)
    ideal_dcg = np.sum((2 ** np.array(ideal_relevance[:k]) - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate NDCG@K
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

    return ndcg


# precisions = []
# recalls = []
# mrrs = []
# ndcgs = []


# Define a function that computes metrics for a single user
def compute_metrics_for_user(user_index, recommender, interaction_matrix, k=5, relevance_threshold=3.0):
    # Get predictions (top-N items) for the user
    predictions = recommender.predict(user_idx=user_index, top_k=k)

    # Extract the item indices from the predictions
    prediction_indexes = np.array([item[0] for item in predictions], dtype=int)

    # Extract the interaction scores for the predicted items
    scores = interaction_matrix[user_index, prediction_indexes].toarray().flatten()

    # Apply the relevance threshold
    relevance = (scores > relevance_threshold).astype(int)

    # Calculate relevant items for recall calculation
    relevant_items = (interaction_matrix[user_index].data > relevance_threshold).sum()

    # Calculate metrics
    precision = precision_at_k(relevance, k)
    recall = recall_at_k(relevance, relevant_items)
    mrr = mean_reciprocal_rank(relevance)
    ndcg = ndcg_at_k(relevance, k)

    return precision, recall, mrr, ndcg

def compute_metrics(interaction_matrix, recommender, k=5, relevance_threshold=3.0):
  n_jobs = -1  # Use all available cores
  results = Parallel(n_jobs=n_jobs)(delayed(compute_metrics_for_user)(
      user_index, recommender, interaction_matrix, k, relevance_threshold
  ) for user_index in tqdm(range(interaction_matrix.shape[0])))

  return results