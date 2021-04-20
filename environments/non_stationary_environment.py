from environments.environment import Environment
import numpy as np


class NonStationaryEnvironment(Environment):

    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.n_phases = len(self.probabilities)
        self.phase_size = horizon / self.n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        self.t += 1
        p = self.probabilities[current_phase][pulled_arm]
        return np.random.binomial(1, p)
