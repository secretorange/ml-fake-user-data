from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler

def build_nmf(interaction_csr_matrix, number_of_components):
    # scaler = MaxAbsScaler()
    # interaction_matrix_masked_normalized = scaler.fit_transform(interaction_matrix_masked)

    n_components = min(number_of_components, min(interaction_csr_matrix.shape[0], interaction_csr_matrix.shape[1]))

    # Apply NMF
    nmf = NMF(n_components=n_components, init='nndsvd', random_state=0, verbose=True)
    W = nmf.fit_transform(interaction_csr_matrix)
    H = nmf.components_

    return (W, H)