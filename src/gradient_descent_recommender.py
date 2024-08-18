import numpy as np
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix

from recommender_base import RecommenderBase

class GradientDescentRecommender(RecommenderBase):
    def __init__(self, number_of_components=2, learning_rate=0.01, epochs=100, batch_size=10000):
        super().__init__()
        self.number_of_components = number_of_components
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Check if a GPU is available
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))

    def build(self, interaction_csr_matrix):

        n_components = min(self.number_of_components, min(interaction_csr_matrix.shape[0], interaction_csr_matrix.shape[1]))

        # Convert the sparse matrix to a TensorFlow SparseTensor
        sparse_tf = tf.sparse.SparseTensor(indices=np.array([interaction_csr_matrix.nonzero()[0], interaction_csr_matrix.nonzero()[1]]).T,
                                            values=interaction_csr_matrix.data.astype(np.float32),  # Ensure the data is float32
                                            dense_shape=interaction_csr_matrix.shape)
        
        # Initialize W and H matrices with random non-negative values
        self.W = tf.Variable(tf.random.normal([interaction_csr_matrix.shape[0], n_components]), dtype=tf.float32)
        self.H = tf.Variable(tf.random.normal([n_components, interaction_csr_matrix.shape[1]]), dtype=tf.float32)

        # Optimization loop
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        for epoch in range(self.epochs):
            for i in range(0, interaction_csr_matrix.shape[0], self.batch_size):
                with tf.GradientTape() as tape:
                    # Perform matrix multiplication on the batch
                    W_batch = self.W[i:i+self.batch_size]
                    WH = tf.matmul(W_batch, self.H)

                    # Extract indices in this batch
                    batch_indices = np.where((sparse_tf.indices[:, 0] >= i) & (sparse_tf.indices[:, 0] < i + self.batch_size))[0]
                    A_batch_values = tf.gather(sparse_tf.values, batch_indices)
                    A_batch_indices = tf.gather(sparse_tf.indices, batch_indices)

                    # Adjust indices for the batch
                    A_batch_indices_adjusted = A_batch_indices - tf.constant([i, 0], dtype=tf.int64)

                    # Extract the corresponding WH values for these indices
                    WH_sparse_values = tf.gather_nd(WH, A_batch_indices_adjusted)

                    # Calculate loss (ensure batch_values and WH_sparse.values have matching shapes)
                    loss = tf.reduce_mean(tf.square(A_batch_values - WH_sparse_values))

                # Compute and apply gradients
                gradients = tape.gradient(loss, [self.W, self.H])
                optimizer.apply_gradients(zip(gradients, [self.W, self.H]))

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

    def predict(self, user_idx, top_k):
         # Compute the predicted ratings for this user using W and H
        user_ratings = np.dot(self.W[user_idx, :], self.H)

        return self._sort(user_ratings, top_k)
 