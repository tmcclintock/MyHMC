"""An implementation of the leapfrog algorithm with
an arbitrary mass matrix.
"""
import numpy as np

class leapfrogger(object):
    def __init__(self, q0, p0, gradU, epsilon, L, Minv=None):
        """An object for performing leapfrog integration.

        Args:
            q0: original coordinates
            p0: original momenta
            gradU (callable): the gradient of the potential
            epsilon: the 'time step'
            L: number of steps to take
            M (optional): mass matrix

        """
        q0 = np.asarray(q0)
        p0 = np.asarray(p0)
        self.q0 = q0
        self.p0 = p0
        self.q = np.zeros((L+1, len(q0)))
        self.q[0] = q0
        self.p = np.zeros_like(self.q)
        self.p[0] = p0
        self.gradU = gradU
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
        gradU_i = gradU_i = self.gradU(self.q[0])
        for i in range(1, self.L+1):

            #Leapfrog update the positions
            self.q[i] = self.q[i-1] \
                + e * Minv @ (self.p[i-1] - 0.5 * e2 * gradU_i)

            #Leapfrog update the momenta
            gradU_i1 = self.gradU(self.q[i])
            self.p[i] = self.p[i-1] \
                - 0.5 * e * (gradU_i1 + gradU_i)

            #Update the force gradient variable
            gradU_i = gradU_i1
        return

if __name__ == "__main__":
    def gradU(q):
        #U(q) = - 1/q
        r = np.sqrt(q.T @ q)
        return q / r**3
        
    q0 = np.array([1, 0])
    p0 = np.array([0, 1])
    e = 0.001
    L = 10000

    LF = leapfrogger(q0, p0, gradU, e, L)
    LF.compute_trajectory()

    import matplotlib.pyplot as plt
    q = LF.q
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(q[:,0], q[:,1])
    plt.show()
