import numpy as np


def canonical_dot_product(xi, xj):
    return sum(xii*xjj for xii,xjj in zip(xi, xj))


def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    e_val = (A.dot(b_k).dot(b_k)) / np.dot(b_k, b_k) #Calculate the eigenvalue of the found eigenvector
    return b_k, e_val



class PCA:
    """
    PCA algorithm as outlined in Mathematics for Machine Learning Book.
    Given a set of data, subspace dimension (M) and a kernel, projection matrix is fit to the data using eigendecomposition.
    Calling the object on data returns the solution projected to the subspace.
    User can obtain the reconstructed solution by calling the recon() method.
    """

    def __init__(self, data, kernel, M):
        self.data = data
        self.kernel = kernel
        if M > data.shape[1]:
            raise Exception("M must be less or equal to solution dimension")
        self.M = M
        self.B = None #Projection Matrix

    def fit(self):
        """
        Fits the project matrix
        :return None:
        """

        # Center the data
        meaned_data = self.data - np.mean(self.data, axis=0)

        #Generate the covariance matrix
        cov_mat = np.asarray([[self.kernel(xi, xj) for xi in dataset.T] for xj in dataset.T])
        self.B = np.zeros(shape=(cov_mat.shape[0], self.M))

        for mi in range(self.M):
            eigenvector, eigenval = power_iteration(cov_mat, 80)
            self.B[:,mi] = eigenvector
            cov_mat = cov_mat - np.outer(eigenvector, eigenvector.T) * (eigenval / np.linalg.norm(eigenvector))



    def __call__(self, x):
        x_meaned = x - np.mean(self.data, axis=0)
        X_reduced = np.dot(self.B.transpose(), x_meaned).transpose()
        X_back = np.dot(self.B, X_reduced.T).T + np.mean(self.data, axis=0)

        return X_reduced, X_back



if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    digits = load_digits()
    datay = np.where(digits.target==2)
    dataset = digits.data[datay]


    plt.gray()
    fig, axs = plt.subplots(3, 2)

    pca_ob = PCA(dataset, canonical_dot_product, 1)
    pca_ob.fit()
    dataset_idx = 10
    
    x_0 = pca_ob(dataset[dataset_idx])[1].reshape((8,8))
    axs[0, 0].matshow(x_0)
    axs[0, 0].set_title("M=1 Eigenvectors used")
    axs[0, 0].set_yticklabels([])
    axs[0, 0].set_xticklabels([])

    pca_ob = PCA(dataset, canonical_dot_product, 10)
    pca_ob.fit()

    x_0 = pca_ob(dataset[dataset_idx])[1].reshape((8, 8))
    axs[0, 1].matshow(x_0)
    axs[0, 1].set_title("M=10 Eigenvectors used")
    axs[0, 1].set_yticklabels([])
    axs[0, 1].set_xticklabels([])

    pca_ob = PCA(dataset, canonical_dot_product, 30)
    pca_ob.fit()

    x_0 = pca_ob(dataset[dataset_idx])[1].reshape((8, 8))
    axs[1, 0].matshow(x_0)
    axs[1, 0].set_title("M=30 Eigenvectors used")
    axs[1, 0].set_yticklabels([])
    axs[1, 0].set_xticklabels([])

    pca_ob = PCA(dataset, canonical_dot_product, 50)
    pca_ob.fit()

    x_0 = pca_ob(dataset[dataset_idx])[1].reshape((8, 8))
    axs[1, 1].matshow(x_0)
    axs[1, 1].set_title("M=50 Eigenvectors used")
    axs[1, 1].set_yticklabels([])
    axs[1, 1].set_xticklabels([])

    pca_ob = PCA(dataset, canonical_dot_product, 60)
    pca_ob.fit()

    x_0 = pca_ob(dataset[dataset_idx])[1].reshape((8, 8))
    axs[2, 0].matshow(x_0)
    axs[2, 0].set_title("M=60 Eigenvectors used")
    axs[2, 0].set_yticklabels([])
    axs[2, 0].set_xticklabels([])

    x_0 = dataset[dataset_idx].reshape((8, 8))
    axs[2, 1].matshow(x_0)
    axs[2, 1].set_title("Original")
    axs[2, 1].set_yticklabels([])
    axs[2, 1].set_xticklabels([])

    plt.show()





