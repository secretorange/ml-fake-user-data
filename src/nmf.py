from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

def build_nmf(interaction_csr_matrix, number_of_components):
    # scaler = MaxAbsScaler()
    # interaction_matrix_masked_normalized = scaler.fit_transform(interaction_matrix_masked)

    n_components = min(number_of_components, min(interaction_csr_matrix.shape[0], interaction_csr_matrix.shape[1]))

    # Apply NMF
    nmf = NMF(n_components=n_components, init='nndsvd', random_state=0, verbose=True)
    W = nmf.fit_transform(interaction_csr_matrix)
    H = nmf.components_

    return (W, H)


def recommend(W, H, user_index, top_n):
    # Compute the predicted ratings for this user using W and H
    user_ratings = np.dot(W[user_index, :], H)

    # Get the indices of the top N recommendations
    top_item_indices = np.argsort(-user_ratings)[:top_n]

    # Map the item indices back to the item IDs
    item_ids = interaction_matrix.indices[top_item_indices]

    # Store the results in a list
    user_recommendations = {
        'user_id': interaction_matrix.indices[user_index],
        'item_id': items_df.iloc[top_item_indices]['item_id'].tolist()
    }

    return pd.DataFrame(user_recommendations)