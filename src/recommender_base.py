import numpy as np

class RecommenderBase:
    def __init__(self):
        # Initialize any necessary attributes
        pass

    def build(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, user_idx):
        """
        This method should be overridden by subclasses.
        It is intended to predict items or ratings for a given user.

        Parameters:
        user_idx (int): The index of the user for whom to make a prediction.

        Returns:
        list: A list of predicted items or ratings for the user.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _sort(self, scores, top_k):
        top_k_indices = np.argsort(-scores)[:top_k]

        # Get the corresponding similarity scores
        top_k_scores = scores[top_k_indices]

        # Combine indices and scores into a list of tuples
        top_k_results = list(zip(top_k_indices, top_k_scores))

        return top_k_results