"""An implementation of the leapfrog algorithm with
an arbitrary mass matrix.
"""

class LeapFrogger(object):
    def __init__(self, q0, p0, force_function, epsilon, N, M==1):
        self.q0 = q0
        self.p0 = p0
        self.force_function = force_function
        self.epsilon = epsilon
        self.N = N
        self.M = M

    def compute_trajectory(self):
        for i in range(self.N):
            break
        return
