import time
import numba
import numpy as np
from numpy.linalg import pinv as np_pinv
import jax.numpy as jnp
from scipy.linalg import pinvh
#from scipy.linalg import pinv2 as scipy_pinv2

@numba.njit
def np_jit_pinv(A):
  return np_pinv(A)

matrix = np.random.rand(100, 100)
for pinv in [np_pinv, np_jit_pinv, jnp.linalg.pinv, np.linalg.inv]:
    start = time.time()
    pinv(matrix)
    print(f'{pinv.__module__ +"."+pinv.__name__} took {time.time()-start:.3f}')

matrix_symm = (matrix + matrix.T)/2
for eig in [np.linalg.eig, np.linalg.eigh]:
    start = time.time()
    a, b = eig(matrix_symm)
    print(np.shape(a))
    print(np.shape(b))
    print(f'{eig.__module__ +"."+eig.__name__} took {time.time()-start:.3f}')

# Takeaways
# For pinv small matrix, use np. For pinv big matrix use jax.
# For pinv symmetric matrix, use scipy pinvh maybe, numpy pinv is still pretty fast.
# np.linalg.inv is so fast though........
# For eig, try out np linalg eigh for symmetric matrices maybe