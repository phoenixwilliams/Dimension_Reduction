# Dimension Reduction Library
### The aim of this library is to contain coded algorithms of dimension reduction algorithms that I study.
#### This library was created primarily for my own eduction, however if this can be of use to anyone else for educational purposes then enjoy!

## - PCA (Principle Component Algorithm)
##### Given a dataset of d-dimensional vectors, this algorithm generates a projection matrix through using M eigenvectors of the dataset covariance matrix.
##### These eigenvectors correspond to the M-largest eigenvalues of the covariance matrix.

##### This implementation applies the power iteration method to compute the eigenvectors (for M-largest eigenvalues) of the covariance matrix.
##### The implementation contains an example of the MNIST digit dataset obtained from sklearn, this example shows the effect the value of M has on reconstructing the data from the lower dimensional subpsace. 