"""My implementation of a Hamiltonian Monte Carlo (HMC) sampler.
"""
import numpy as np

class MyHMC(class):
    def __init__(self, lnpost):
        self.lnpost = lnpost
