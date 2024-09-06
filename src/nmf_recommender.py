from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

from recommender_base import RecommenderBase

class NMFRecommender(RecommenderBase):
    def __init__(self, number_of_components):
        super().__init__()
        self.number_of_components = number_of_components

    def build(self, interaction_csr_matrix):
        n_components = min(self.number_of_components, min(interaction_csr_matrix.shape[0], interaction_csr_matrix.shape[1]))

        nmf = NMF(n_components=n_components, init='nndsvd', random_state=0, verbose=True)
        self.W = nmf.fit_transform(interaction_csr_matrix)
        self.H = nmf.components_

    def predict(self, user_idx, top_k, item_indices=None):
        # Compute the predicted ratings for this user using W and H
        user_scores = np.dot(self.W[user_idx, :], self.H)

        if item_indices == None:
            return self._sort(user_scores, top_k)
        else:
            return self._prepare(self, user_scores, item_indices)