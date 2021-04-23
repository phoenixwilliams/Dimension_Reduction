import numpy as np

def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q


class SVD:

    def __init__(self, data):
        self.S = None
        self.U = None
        self.V = None
        self.data = data

    def fit(self):
        A = self.data
        ata = A.T.dot(A)
        eigenvals, eigvectors = np.linalg.eig(ata)
        ordered_eigenvals = -np.sort(-eigenvals)
        self.S = np.zeros(A.shape)
        np.fill_diagonal(self.S, np.sqrt(ordered_eigenvals))
        Sinv = np.linalg.inv(self.S)

        idxs = np.where(ordered_eigenvals == eigenvals)[0]
        self.V = eigvectors[idxs]
        self.U = A.dot(self.V).dot(Sinv)

    def u_s_vt(self):
        return self.U, self.S, self.V.T


if __name__ == "__main__":

    A = np.asarray([[4, 0], [3, -5]])

    svd = SVD(A)
    svd.fit()

    U, S, VT = svd.u_s_vt()

    print(U.dot(S).dot(VT))






