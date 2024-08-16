import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_user_similarity(user_id, user_profiles, item_features):
    # Get the user profile (as a dense array if it's not already)
    user_profile = user_profiles.loc[user_id].to_numpy().reshape(1, -1)  # Ensure it's a 2D array

    # Calculate the cosine similarity between this user's profile and all item profiles
    similarity_scores = cosine_similarity(user_profile, item_features.to_numpy())

    # Convert the similarity scores to a Series for easier interpretation
    similarity_series = pd.Series(similarity_scores.flatten(), index=item_features.index)

    # Optionally, sort items by similarity score
    similarity_series_sorted = similarity_series.sort_values(ascending=False)

    df = pd.DataFrame(similarity_series_sorted)

    df.columns = ['score']

    return df