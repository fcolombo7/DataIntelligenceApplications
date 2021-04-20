import numpy as np

from learners.Learner import Learner


class Greedy(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_payoffs = np.zeros(self.n_arms)

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        best_arms = np.argwhere(self.expected_payoffs == self.expected_payoffs.max()).reshape(-1)
        return np.random.choice(best_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # compute the mean in an incremental way
        self.expected_payoffs[pulled_arm] = (self.expected_payoffs[pulled_arm] * (self.t - 1) + reward) / self.t
