import numpy as np
from learners.Learner import Learner


class UCB(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(self.n_arms)
        #self.confidences = np.array([np.inf]*n_arms)
        self.confidences = np.zeros(self.n_arms)

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        upper_bounds = self.empirical_means + self.confidences
        best_arms = np.argwhere(upper_bounds == upper_bounds.max()).reshape(-1)
        return np.random.choice(best_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        # compute the mean in an incremental way
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t
        # update all the upper confidence bounds
        for arm in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[arm])
            self.confidences[arm] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        # at the denominator there is N_a @ t-1, so append after the computation of the confidence.
        self.update_observations(pulled_arm, reward)
