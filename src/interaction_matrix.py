# NEW
import numpy as np
import pandas as pd
import time
from typing import ValuesView
from tqdm import tqdm
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix, save_npz
 
def log_scale_with_threshold(input_values, start_threshold, value_of_x_when_y_is_1=1, log_base=2):
  scale_factor = log_base  # Now adjustable based on log_base
  shift_constant = (value_of_x_when_y_is_1 - start_threshold * scale_factor) / (scale_factor - 1)
  normalization_constant = np.log(start_threshold + shift_constant) / np.log(log_base)
  return np.log(input_values + shift_constant) / np.log(log_base) - normalization_constant

# Log scale function
def log_scale(x):
    return np.log2(x) + 1

# Time decay function
def time_decay(timestamp, current_time, decay_rate=1e-8):
    return np.exp(-decay_rate * (current_time - timestamp))

DEBUG = False

def debug(text):
  if DEBUG == True:
    print(text)

# def build_interaction_matrix(interaction_config, items_df, users_df, interactions_df):
#   # Current time
#   current_time = time.time()

#   # Compute time decay vector for all timestamps
#   print('Timestamp Decayed')
#   interactions_df['timestamp_decayed'] = time_decay(interactions_df['timestamp'].values, current_time)

#   # Map interaction types to their weights
#   print('Interaction Weights')
#   interactions_df['interaction_weight'] = interactions_df['type'].map(lambda x: interaction_config[x]['weight'])

#   # Compute the decayed weights
#   print('Decayed Weights')
#   interactions_df['weighted_value'] = interactions_df['interaction_weight'] * interactions_df['timestamp_decayed']

#   # Group by user_id, item_id, and type, and sum the values
#   print('Sums')
#   interaction_sums = interactions_df.groupby(['user_id', 'item_id', 'type']).agg({
#       'value': 'sum',
#       'weighted_value': 'sum'
#   }).reset_index()

#   # # Apply log scale and any other adjustments


#   # Create a new column to store the thresholds
#   print('Thresholds')
#   interaction_sums['threshold'] = interaction_sums['type'].map(lambda x: interaction_config[x]['threshold'])

#   # Vectorized application of log scale with or without threshold
#   print('Final Value')
#   interaction_sums['final_value'] = np.where(
#       interaction_sums['threshold'].notna(),  # Check if the threshold is not None
#       log_scale_with_threshold(interaction_sums['value'], interaction_sums['threshold']),  # Apply log scale with threshold
#       log_scale(interaction_sums['value'])  # Apply regular log scale
#   )

#   # Multiply by the precomputed weighted value
#   print('Final Weighted Value')
#   interaction_sums['final_weighted_value'] = interaction_sums['final_value'] * interaction_sums['weighted_value']

#   # Initialize sparse matrix using scipy's dok_matrix
#   print('Init Sparse Matrix')
#   interaction_matrix = dok_matrix((len(users_df), len(items_df)), dtype=np.float32)

#   # Map user_id and item_id to indices
#   print('Map user_id and item_id')
#   user_map = {user_id: idx for idx, user_id in enumerate(users_df['user_id'])}
#   item_map = {item_id: idx for idx, item_id in enumerate(items_df['item_id'])}

#   # Convert the DataFrame columns to NumPy arrays
#   user_ids = interaction_sums['user_id'].values
#   item_ids = interaction_sums['item_id'].values
#   final_values = (interaction_sums['final_weighted_value'] + 3).values.astype(np.float32)

#   # Map user_ids and item_ids to their respective indices
#   user_indices = np.array([user_map[user_id] for user_id in user_ids])
#   item_indices = np.array([item_map[item_id] for item_id in item_ids])

#   # Collect the updates into COO format
#   print('Create Upddate Matrix')
#   update_matrix = coo_matrix((final_values, (user_indices, item_indices)), shape=interaction_matrix.shape)

#   print('Convert Interation Matric to CSR Matrix')
#   interaction_matrix = interaction_matrix.tocsr()

#   # Apply the updates to the original sparse matrix
#   print('Apply Update to Interaction Matrix')
#   interaction_matrix += update_matrix

#   return interaction_matrix

class InteractionMatrixBuilder:
    def __init__(self, interaction_config, user_col='user_id', item_col='item_id'):
        self.user_col = user_col
        self.item_col = item_col
        self.interaction_config = interaction_config

    def build(self, interactions_df):
        # Create mappings from user_id and item_id to row and column indices
        user_to_index = {user_id: idx for idx, user_id in enumerate(interactions_df[self.user_col].unique())}
        index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
        
        item_to_index = {item_id: idx for idx, item_id in enumerate(interactions_df[self.item_col].unique())}
        index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

        # Current time
        current_time = time.time()

        # Compute time decay vector for all timestamps
        print('Timestamp Decayed')
        interactions_df['timestamp_decayed'] = time_decay(interactions_df['timestamp'].values, current_time)

        # Map interaction types to their weights
        print('Interaction Weights')
        interactions_df['interaction_weight'] = interactions_df['type'].map(lambda x: self.interaction_config[x]['weight'])

        # Compute the decayed weights
        print('Decayed Weights')
        interactions_df['weighted_value'] = interactions_df['interaction_weight'] * interactions_df['timestamp_decayed']

        # Group by user_id, item_id, and type, and sum the values
        print('Sums')
        interaction_sums = interactions_df.groupby(['user_id', 'item_id', 'type']).agg({
            'value': 'sum',
            'weighted_value': 'sum'
        }).reset_index()

        # # Apply log scale and any other adjustments


        # Create a new column to store the thresholds
        print('Thresholds')
        interaction_sums['threshold'] = interaction_sums['type'].map(lambda x: self.interaction_config[x]['threshold'])

        # Vectorized application of log scale with or without threshold
        print('Final Value')
        interaction_sums['final_value'] = np.where(
            interaction_sums['threshold'].notna(),  # Check if the threshold is not None
            log_scale_with_threshold(interaction_sums['value'], interaction_sums['threshold']),  # Apply log scale with threshold
            log_scale(interaction_sums['value'])  # Apply regular log scale
        )

        # Multiply by the precomputed weighted value
        print('Final Weighted Value')
        interaction_sums['final_weighted_value'] = interaction_sums['final_value'] * interaction_sums['weighted_value']

        # Initialize sparse matrix using scipy's dok_matrix
        print('Init Sparse Matrix')
        interaction_matrix = dok_matrix((len(user_to_index), len(item_to_index)), dtype=np.float32)

        # Map user_id and item_id to indices
        print('Map user_id and item_id')
        
        # Convert the DataFrame columns to NumPy arrays
        user_ids = interaction_sums['user_id'].values
        item_ids = interaction_sums['item_id'].values
        final_values = (interaction_sums['final_weighted_value'] + 3).values.astype(np.float32)

        # Map user_ids and item_ids to their respective indices
        user_indices = np.array([user_to_index[user_id] for user_id in user_ids])
        item_indices = np.array([item_to_index[item_id] for item_id in item_ids])

        # Collect the updates into COO format
        print('Create Upddate Matrix')
        update_matrix = coo_matrix((final_values, (user_indices, item_indices)), shape=interaction_matrix.shape)

        print('Convert Interation Matric to CSR Matrix')
        interaction_matrix = interaction_matrix.tocsr()

        # Apply the updates to the original sparse matrix
        print('Apply Update to Interaction Matrix')
        interaction_matrix += update_matrix

        # Return both the matrix and the mappings
        return interaction_matrix, {'user_to_index': user_to_index, 'index_to_user': index_to_user,
                                    'item_to_index': item_to_index, 'index_to_item': index_to_item}