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

# Function to compute similarity between two items (using item features or precomputed matrix)
def item_similarity(item1, item2, similarity_matrix=None):
    if similarity_matrix is not None:
        return similarity_matrix[item1, item2]
    else:
        # If no similarity matrix is provided, use a dummy similarity (e.g., 0)
        return 0

def calculate_item_popularity(interaction_matrix):
    """
    Calculate the popularity of each item based on user-item interactions.

    Parameters:
    interaction_matrix (scipy.sparse matrix): The user-item interaction matrix.

    Returns:
    dict: A dictionary where keys are item indices and values are the popularity scores (e.g., counts).
    """

    # Sum the interaction matrix along the rows (axis=0) to get the count of interactions per item
    # Convert the resulting array to a flat list and enumerate to create a dictionary
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
    
    # Convert to dictionary: item index as key and its popularity as value
    item_popularity_dict = {item_index: popularity for item_index, popularity in enumerate(item_popularity)}

    return item_popularity_dict

# Function to compute intra-list diversity (ILD)
def intra_list_diversity(prediction_indexes, similarity_matrix=None):
    if len(prediction_indexes) <= 1:
        return 0.0
    diversity = 0.0
    count = 0
    for i in range(len(prediction_indexes)):
        for j in range(i + 1, len(prediction_indexes)):
            similarity = item_similarity(prediction_indexes[i], prediction_indexes[j], similarity_matrix)
            diversity += (1 - similarity)
            count += 1
    return diversity / count if count > 0 else 0.0

# Function to compute novelty based on item popularity
def novelty(prediction_indexes, item_popularity):
    if len(prediction_indexes) == 0:
        return 0.0
    novelty_score = 0.0
    for item in prediction_indexes:
        novelty_score += -np.log(item_popularity.get(item, 1) / len(item_popularity))
    return novelty_score / len(prediction_indexes)


# Define a function that computes metrics for a single user
def compute_metrics_for_user(user_index, recommender, interaction_matrix, item_popularity, similarity_matrix=None, k=5, relevance_threshold=3.0):
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

    # Calculate diversity and novelty
    diversity = intra_list_diversity(prediction_indexes, similarity_matrix)
    novelty_score = novelty(prediction_indexes, item_popularity)

    return precision, recall, mrr, ndcg, diversity, novelty_score

def compute_metrics(interaction_matrix, recommender, k=5, relevance_threshold=3.0):    
    # Calculate item popularity outside the loop
    item_popularity = calculate_item_popularity(interaction_matrix)

    # Initialize an empty list to store results
    results = []

    # Loop through each user index in the interaction matrix
    for user_index in tqdm(range(interaction_matrix.shape[0]), position=0, leave=True):
        # Compute metrics for the current user
        user_metrics = compute_metrics_for_user(
            user_index, recommender, interaction_matrix, item_popularity, None, k, relevance_threshold
        )
        
        # Append the computed metrics to the results list
        results.append(user_metrics)

    return results