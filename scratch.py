import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as la
import scipy.sparse.linalg as sla

class DenseLaplacian(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def eigenvalues(self, k=None, which='SA'):
        return la.eigvalsh(self)
        
    def eigenvectors(self, k=None, which='SA'):
        return la.eigh(self)

class SparseLaplacian(csr_matrix):
    def eigenvalues(self, k=6, which='SA'):
        if k >= self.shape[0]:
            return la.eigvalsh(self.toarray())
        evals, _ = sla.eigsh(self.astype(float), k=k, which=which)
        return evals
        
    def eigenvectors(self, k=6, which='SA'):
        if k >= self.shape[0]:
            return la.eigh(self.toarray())
        return sla.eigsh(self.astype(float), k=k, which=which)

# test dense
Ld = DenseLaplacian(np.array([[2, -1], [-1, 2]]))
print(Ld.eigenvalues())
print(Ld + Ld) # checking math works and what it returns

# test sparse
Ls = SparseLaplacian(csr_matrix(np.array([[2, -1], [-1, 2]])))
print(Ls.eigenvalues(k=1))
print(Ls + Ls) # checking math works
