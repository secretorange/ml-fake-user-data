def pick_recommendations(predictions, items_df):
    # Extract the indexes and similarity scores from predictions
    indexes, similarity_scores = zip(*predictions)

    # Select the relevant rows from items_df using the indexes
    recommendations = items_df.loc[list(indexes)].copy()

    # Add the similarity scores as a new column
    recommendations['similarity_score'] = similarity_scores

    return recommendations