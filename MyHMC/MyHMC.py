"""My implementation of a Hamiltonian Monte Carlo (HMC) sampler.
"""
import numpy as np

class MyHMC(class):
    def __init__(self, lnpost, q0, Nsamples, M):
        self.lnpost = lnpost
        self.q0 = q0
        self.Nsamples = Nsamples
        self.M = M
        self.Minv = np.linalg.inv(M)

        #Write a method to sample the momenta
        #Make a leapfrogger
        #Keep track of samples
        #Loop over the leapfrogger passing in new samples and
        #momenta every time
        #Write a MH acceptance step
