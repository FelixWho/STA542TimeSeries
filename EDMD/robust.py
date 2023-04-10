'''Hua et al. kernel trick EDMD combined with Sinha et al. Online Robust Koopman https://arxiv.org/pdf/2212.05259.pdf'''

import math
import numpy as np
from scipy.spatial import distance


class RobustEDMD:
    '''
    Please be mindful of the sampling rate. 
    Too fast of a rate might not give RobustEDMD enough time to update Koopman operator.
    '''
    def __init__(self, M, delta=0.5, sigma_2=100):
        '''
        M: data window
        delta: regularization parameter
        sigma_2: RBF kernel width
        '''
        self.A = np.zeros((M, M)) # initialize as M x M 0-matrix
        self.G_hat_inv = delta * np.identity(M) # initialize as delta * identity matrix
        self.K = self.G_hat_inv @ self.A # This should just equal the M x M 0-matrix, but whatever
        self.sigma_2 = sigma_2

    def update_koopman_and_forecast_point(self, new_data):
        # Kernel trick
        phi_t_phi = np.exp(-1/self.sigma_2 * distance.squareform(distance.pdist(new_data)))

        # Compute the denominator of G_hat_inv_m+1
        denom = 1 + np.sum(np.transpose(phi_t_phi) @ self.G_hat_inv)

        # Compute updated G_hat_inv_m --> G_hat_inv_m+1
        self.G_hat_inv -= (1/denom) * (self.G_hat_inv @ phi_t_phi @ self.G_hat_inv)
        
        # Compute updated A_m --> A_m+1
        self.A += phi_t_phi
        
        # Update Koopman operator K_m --> K_m+1
        self.K = self.G_hat_inv @ self.A
        
        # Forecast new data point
        



    

def main():
    pass


if __name__ == '__main__':
    main()