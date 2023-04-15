'''
Hua et al. kernel trick EDMD combined with Sinha et al. Online Robust Koopman https://arxiv.org/pdf/2212.05259.pdf
Update: I think this is a better rendition of online Koopman, without regularization parameter: https://arxiv.org/pdf/1909.12520.pdf
'''

import math
import numpy as np
from scipy.spatial import distance
from collections import defaultdict


class RobustEDMD:
    '''
    Please be mindful of the sampling rate. 
    Too fast of a rate might not give RobustEDMD enough time to update Koopman operator.
    Might want to keep M < 100
    '''
    def __init__(self, M, delta=0.5, sigma_2=100, extension_length=50):
        '''
        M: data window, X = [x_1, ..., x_M]
        delta: regularization parameter
        sigma_2: RBF kernel width, default 100 according to HT paper
        '''
        self.A = np.zeros((M, M)) # initialize as M x M 0-matrix
        self.G_hat = delta * np.identity(M) # initialize as delta * identity matrix
        self.G_hat_inv = 1/delta * np.identity(M) # initialize as 1/delta * identity matrix
        self.K = self.G_hat_inv @ self.A # This should just equal the M x M 0-matrix, but whatever
        self.sigma_2 = sigma_2
        self.M = M
        self.extention_length = extension_length
        self.delta = delta

        self.timestep = 0
        self.ready = False

        self.forecasts = defaultdict(float) # Map timestamp --> forecasted value

    def initialize(self, xp: np.ndarray, yp: np.ndarray):
        """
        Let's try not to use this...
        Initialize online EDMD with first p (p >= self.M) snapshot pairs stored in (xp, yp)
        Args:
            xp (np.ndarray): 2D array, shape (self.M, p), matrix [x(1),x(2),...x(p)]
            yp (np.ndarray): 2D array, shape (self.M, p), matrix [y(1),y(2),...y(p)]
        """
        assert xp.shape == yp.shape
        p = xp.shape[1]
        
        Uxy = np.vstack([xp, yp])

        tmp = distance.pdist(Uxy, 'sqeuclidean')
        
        Uga = np.exp(-1/self.sigma_2 * distance.squareform(tmp))
        Uga = Uga[:, 0:self.M]

        self.G_hat = Uga[0:self.M, :]
        self.A = Uga[self.M: , :self.M]

        self.G_hat_inv = np.linalg.pinv(self.G_hat + np.diag(np.full(self.M, self.delta)))

        self.K = self.G_hat_inv @ self.A

        self.timestep += p

        if self.timestep >= 2 * self.M:
            self.ready = True


    def update_koopman_and_forecast_point(self, x, y):
        '''
        x is column vector with shape M x 1: [x1, ..., xM]
        y is column vector with shape M x 1: [x2, ..., xM+1]
        Uxy has shape 2M x 1
        tmp has shape 2M x 2M
        phix_phix_t/phiy_phix_t have shape M x M
        self.G_hat/self.G_hat_inv has shape M x M
        self.A has shape M x M
        '''

        assert x.shape[0] == self.M
        assert y.shape == x.shape

        Uxy = np.vstack([x, y])

        # Kernel trick
        tmp = np.exp(-1/self.sigma_2 * distance.squareform(distance.pdist(Uxy, 'sqeuclidean')))

        phix_phix_t = tmp[:self.M, :self.M]
        phiy_phix_t = tmp[self.M:, :self.M]

        # Based off Sinha et al.
        # Compute the denominator of G_hat_inv_m+1
        denom = 1 + np.sum(np.transpose(phix_phix_t) @ self.G_hat_inv)

        # Compute updated G_hat_m --> G_hat_m+1
        self.G_hat += phix_phix_t
        
        # Compute updated G_hat_inv_m --> G_hat_inv_m+1
        self.G_hat_inv -= (1/denom) * (self.G_hat_inv @ phix_phix_t @ self.G_hat_inv)
        
        # Compute updated A_m --> A_m+1
        self.A += phiy_phix_t
        
        # Update Koopman operator K_m --> K_m+1
        self.K = self.G_hat_inv @ self.A

        self.timestep += 1
        if self.timestep >= 2 * self.M:
            self.ready = True

        if self.ready:
            # Forecast new data point using updated Koopman operator
            # Based off Hua et al.
            mu, Vhat = np.linalg.eig(self.K) # mu is koopman eigenvalues

            # We need to find Sigma2, Q = eig(G_hat), but we only have G_hat_inv <--- EDIT: we do actually track G_hat, my bad
            # Linear algebra fact: for every eigenvector, eigenvalue pair (v, lambda) of G_hat,
            # (v, 1/lambda) is an eigenvector, eigenvalue pair of G_hat_inv
            Sigma2, Q = np.linalg.eig(self.G_hat_inv)
            SigmaPINV = np.sqrt(Sigma2)

            # Get the koopman eigenfunctions
            Uga = np.vstack([self.G_hat, self.A])

            Phixy = Uga @ np.multiply(Q, SigmaPINV) @ Vhat

            # Get the koopman modes
            xi = np.linalg.pinv(Phixy) @ Uxy

            forecast = self.forecast_point(xi, mu, Phixy[-1, :])
            self.forecasts[self.timestep + self.extention_length] = forecast.item()
            print(forecast)

    def forecast_point(self, xi, mu, phi_end):
        '''
        Forecast single point self.extension_length out into the future
        '''

        print(np.isnan(xi).any(), np.isnan(mu).any(), np.isnan(phi_end).any())

        tmp = phi_end.T
        print(tmp.shape, mu.shape, xi.T.shape)
        for kk in range(self.extention_length):
            tmp = mu * tmp

        forecast = np.real(xi.T @ tmp)
        return forecast


def main():
    pass


if __name__ == '__main__':
    main()