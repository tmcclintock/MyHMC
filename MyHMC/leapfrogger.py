"""An implementation of the leapfrog algorithm with
an arbitrary mass matrix.
"""
import numpy as np

class leapfrogger(object):
    def __init__(self, q0, p0, neg_grad_U, epsilon, L, Minv=None, save_trajectory=False):
        """An object for performing leapfrog integration.

        Args:
            q0: original coordinates
            p0: original momenta
            neg_grad_U (callable): negative of the gradient of the potential
            epsilon: the 'time step'
            L: number of steps to take
            M (optional): mass matrix
            save_trajectory (optional): flag for whether or not to
                save the entire trajectory or just evolve the system

        """
        q0 = np.asarray(q0)
        p0 = np.asarray(p0)
        self.q0 = q0
        self.p0 = p0
        self.save_trajectory = save_trajectory
        if self.save_trajectory:
            self.q = np.zeros((L+1, len(q0)))
        else:
            self.q = np.zeros((2, len(q0)))
        self.q[0] = q0
        self.p = np.zeros_like(self.q)
        self.p[0] = p0
        self.neg_grad_U = neg_grad_U
        self.epsilon = epsilon
        self.L = L
        if Minv is None:
            self.Minv = np.eye(len(q0))
        else:
            self.Minv = Minv

    def compute_trajectory(self):
        """Compute the trajectory.
        """
        e = self.epsilon
        e2 = e**2
        L = self.L
        Minv = self.Minv

        #Compute the forces once each step
        ngradU_i = self.neg_grad_U(self.q[0])
        for i in range(1, self.L+1):

            if self.save_trajectory:
                #Leapfrog update the positions
                self.q[i] = self.q[i-1] \
                    + e * Minv @ (self.p[i-1] - 0.5 * e2 * ngradU_i)
                #Leapfrog update the momenta
                ngradU_i1 = self.neg_grad_U(self.q[i])
                self.p[i] = self.p[i-1] \
                    - 0.5 * e * (ngradU_i1 + ngradU_i)
            else:
                #Leapfrog update the positions
                self.q[1] = self.q[0] \
                    + e * Minv @ (self.p[0] - 0.5 * e2 * ngradU_i)
                #Leapfrog update the momenta
                ngradU_i1 = self.neg_grad_U(self.q[1])
                self.p[1] = self.p[0] \
                    - 0.5 * e * (ngradU_i1 + ngradU_i)
                self.q[0] = self.q[1]
                self.p[0] = self.p[1]

            #Update the force gradient variable
            ngradU_i = ngradU_i1
        return

if __name__ == "__main__":
    def neg_grad_U(q): #negative the log of a Gaussian
        return q
        
    q0 = np.array([1, 0])
    p0 = np.array([0, 1.1])
    e = 0.001
    L = 10000

    LF = leapfrogger(q0, p0, neg_grad_U, e, L, save_trajectory=True)
    LF.compute_trajectory()

    import matplotlib.pyplot as plt
    q = LF.q
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(q[:,0], q[:,1])
    plt.show()
