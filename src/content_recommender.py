import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recommender_base import RecommenderBase

class ContentRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()

    def build(self, interaction_csr_matrix, item_features):
        self.item_features = item_features
        self.user_profiles = interaction_csr_matrix.dot(item_features)

        #return pd.DataFrame(user_profiles, columns=item_features.columns)

    def predict(self, user_idx, top_k, item_indices=None):
        # Get the user profile (as a dense array if it's not already)
        user_profile = self.user_profiles[user_idx].reshape(1, -1)  # Ensure it's a 2D array

        # Calculate the cosine similarity between this user's profile and all item profiles
        similarity_scores = cosine_similarity(user_profile, self.item_features).flatten()

        if item_indices is None:
            return self._sort(similarity_scores, top_k)
        else:
            return self._prepare(similarity_scores, item_indices)
        
        # # Get the indices of the top K recommendations
        # top_k_indices = np.argsort(-similarity_scores)[:top_k]

        # # Get the corresponding similarity scores
        # top_k_scores = similarity_scores[top_k_indices]

        # # Combine indices and scores into a list of tuples
        # top_k_results = list(zip(top_k_indices, top_k_scores))

        # return top_k_results

# def calculate_user_similarity(user_index, user_profiles, item_features):
#     # Get the user profile (as a dense array if it's not already)
#     user_profile = user_profiles.loc[user_index].to_numpy().reshape(1, -1)  # Ensure it's a 2D array

#     # Calculate the cosine similarity between this user's profile and all item profiles
#     similarity_scores = cosine_similarity(user_profile, item_features.to_numpy())

#     # Convert the similarity scores to a Series for easier interpretation
#     similarity_series = pd.Series(similarity_scores.flatten(), index=item_features.index)

#     # Optionally, sort items by similarity score
#     similarity_series_sorted = similarity_series.sort_values(ascending=False)

#     df = pd.DataFrame(similarity_series_sorted)

#     df.columns = ['score']

#     return df

# def build_user_profiles(interaction_csr_matrix, item_features):
#     user_profiles = interaction_csr_matrix.dot(item_features.values)

#     return pd.DataFrame(user_profiles, columns=item_features.columns)
