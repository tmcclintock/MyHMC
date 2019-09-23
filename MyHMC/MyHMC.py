"""My implementation of a Hamiltonian Monte Carlo (HMC) sampler.
"""
import numpy as np
import scipy.stats as ss
import leapfrogger as LF

class MyHMC(object):
    
    def __init__(self, lnpost, grad_lnpost, ndim, M):
        self.lnpost = lnpost
        self.grad_lnpost = grad_lnpost
        self.ndim = ndim
        self.M = M
        self.detM = np.linalg.det(M)
        self.Minv = np.linalg.inv(M)

    def _compute_Hamiltonian(self, q, p):
        d = self.ndim
        M = self.M
        detM = self.detM
        Minv = self.Minv
        Left = -self.lnpost(q)
        right = 0.5 * (d * np.log(2 * np.pi * detM) +
                       p.T @ Minv @ p)
        #print("here", Left, right)
        return Left + right

    def sample_once(self):
        i = self.count
        qi = self.q[i]
        pi = self.p[i]
        H = self._compute_Hamiltonian(qi, pi)


        #Set the leapfrogger step size and number of steps
        #epsilon is set _roughly_ adaptively according to the minimum
        #characteristic time
        #L should be set as the length of the circle around the PDF at
        #the 1 sigma level
        #These choices would get eliminated for NUTS
        epsilon = 10**(np.random.rand()*2 - 3.)
        L = np.random.randint(100, 1000)
        #self.L = int(np.pi/self.epsilon)
        #print(self.epsilon, self.L)
        
        frogger = LF.leapfrogger(qi, pi, self.grad_lnpost,
                                 epsilon, L, Minv = self.Minv)
        frogger.compute_trajectory()
        qnew = frogger.q[-1]
        pnew = frogger.p[-1]
        Hnew = self._compute_Hamiltonian(qnew, pnew)

        ln_r = np.log(np.random.rand())
        #print(Hnew, H)
        upper_limit = np.min([0, -Hnew + H])
        if ln_r < upper_limit: #Accept proposal
            self.count += 1
            self.q[self.count] = qnew
            self.p[self.count] = pnew
        else: #Reject
            self.count += 1
            self.q[self.count] = qi
            self.p[self.count] = pi

        return

    def run(self, q0, Nsamples):
        q0 = np.asarray(q0)
        
        assert len(q0) == self.ndim
        assert len(q0.shape) < 2

        self.Nsamples = Nsamples
        
        self.q = np.zeros((Nsamples, self.ndim))
        self.p = np.zeros((Nsamples, self.ndim))

        self.q0 = q0
        self.p0 = ss.multivariate_normal.rvs(mean = np.zeros_like(q0), cov = M)
        self.q[0] = self.q0
        self.p[0] = self.p0

        #Current number of steps
        self.count = 0

        for i in range(Nsamples - 1):
            self.sample_once()
        return

if __name__ == "__main__":
    def lnpost(q): #Unit normal dist
        return -q.T @ q / 2.
    
    def grad_lnpost(q):
        return -q

    q0 = np.array([0.5, 0.03])
    M = np.eye(2)
    hmc = MyHMC(lnpost, grad_lnpost, len(q0), M)

    Nsamples = 10000

    hmc.run(q0, Nsamples)

    import matplotlib.pyplot as plt
    q = hmc.q
    plt.scatter(q[:, 0], q[:, 1], marker='.')
    plt.show()
