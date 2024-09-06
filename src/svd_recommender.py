from scipy.sparse.linalg import svds
import numpy as np
from recommender_base import RecommenderBase

class SVDRecommender(RecommenderBase):
    def __init__(self, number_of_components=2):
        super().__init__()
        self.number_of_components = number_of_components
        self.U = None
        self.sigma = None
        self.Vt = None

    def build(self, interaction_csr_matrix):
        n_components = min(self.number_of_components, min(interaction_csr_matrix.shape[0], interaction_csr_matrix.shape[1]))

        # Perform SVD on the interaction matrix
        self.U, self.sigma, self.Vt = svds(interaction_csr_matrix, k=n_components)

        # Convert sigma into a diagonal matrix for easy multiplication
        self.sigma = np.diag(self.sigma)

    def predict(self, user_idx, top_k, item_indices=None):    
        # Calculate the user-specific vector by multiplying the U matrix row with sigma
        user_vector = np.dot(self.U[user_idx], self.sigma)

        # Calculate the score for each item by multiplying the user vector with the Vt matrix
        user_scores = np.dot(user_vector, self.Vt)

        if item_indices is None:
            return self._sort(user_scores, top_k)
        else:
            return self._prepare(self, user_scores, item_indices)