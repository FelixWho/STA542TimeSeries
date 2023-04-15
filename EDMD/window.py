'''Hua et al. kernel EDMD, based on https://github.com/AdMeynard/BoundaryEffectsReduction/blob/master/Algorithm/approxKoopman.m'''

import math
import numpy as np
from scipy.spatial import distance


def approximate_koopman(x, y, sigma_2):
    '''
    Evaluate Koopman modes, eigen values, and eigen function (from paper of Hua et al.)
    Input:
       x: input dataset
       y: output dataset
       sigma_2: shape parameter
      
    Output:
       xi: Koopman modes
       mu: Koopman eigen values
       phi_end: Koopman eigen functions
    '''

    m = np.size(x, 0)
    
    Uxy = np.vstack([x, y[-1, :]])

    tmp = distance.pdist(Uxy, 'sqeuclidean')
    
    Uga = np.exp(-1/sigma_2 * distance.squareform(tmp))
    Uga = Uga[:, 0:m]

    Ghat = Uga[0:m, :]
    Ahat = Uga[1:(m+1), :]

    # Unlike Matlab, sigma2 returned by np.linalg.eig() should be a vector instead of a matrix
    sigma2, Q = np.linalg.eig(Ghat)

    SigmaPINV = 1./np.sqrt(sigma2)

    Mtmp = np.multiply(Q, SigmaPINV) # <==> Q * np.transpose(SigmaPINV) since * is element-wise I think

    KoopMat = np.transpose(Mtmp) @ Ahat @ Mtmp

    mu, Vhat = np.linalg.eig(KoopMat)

    Phixy = Uga @ Mtmp @ Vhat

    xi = np.linalg.pinv(Phixy) @ Uxy

    phi_end = Phixy[-1, :]

    return xi, mu, phi_end


def main():
    pass


if __name__ == '__main__':
    main()