from implicit.als import AlternatingLeastSquares
import numpy as np

from recommender_base import RecommenderBase

class ALSRecommender(RecommenderBase):
    def __init__(self, number_of_components=2, regularization=0.1, iterations=20):
        super().__init__()
        self.number_of_components = number_of_components
        self.regularization = regularization
        self.iterations = iterations


    def build(self, interaction_csr_matrix):
        n_components = min(self.number_of_components, min(interaction_csr_matrix.shape[0], interaction_csr_matrix.shape[1]))

        # Initialize the ALS model
        self.model = AlternatingLeastSquares(factors=n_components, regularization=self.regularization, iterations=self.iterations)

        self.model.fit(interaction_csr_matrix)

    def predict(self, user_idx, top_k, item_indices=None):
        if item_indices is None:
            # Get the top K recommendations for the user
            top_k_results = self.model.recommend(user_idx, user_items=[], N=top_k, filter_already_liked_items=False)

            # top_k_results is a tuple of (item_indices, scores)
            top_k_indices, top_k_scores = top_k_results
        else:
            # Score the specific items for the user using rank_items
            top_k_results = self.model.rank_items(user_idx, user_items=[], selected_items=item_indices)
            # top_k_results is a tuple of (item_indices, scores)
            top_k_indices, top_k_scores = zip(*top_k_results)

        # Combine indices and scores into a list of tuples and return
        return list(zip(top_k_indices, top_k_scores))