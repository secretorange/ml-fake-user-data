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

    def predict(self, user_idx, top_k):
        top_k_results = self.model.recommend(user_idx, self.interaction_csr_matrix[user_idx], N=top_k)

        return top_k_results
 