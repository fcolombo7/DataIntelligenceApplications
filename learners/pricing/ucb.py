from learners.pricing.learner import Learner
import numpy as np


class UCB(Learner):
    LEARNER_NAME = 'UCB-Pricing'

    def __init__(self, arm_values, period=30, next_purchases_update='binomial'):
        super().__init__(arm_values, period, next_purchases_update)
        # conv_rate estimation
        self.empirical_means = np.zeros(self.n_arms)
        self.confidences = np.array([np.inf] * self.n_arms)

    def pull_arm(self) -> int:
        # now multiply the upper bound with the arm values! that is known
        upper_bounds = self.empirical_means + self.confidences
        # period * estimation = mean of the binomial
        conditions = self.arm_values * (1 + self.period * self.next_purchases_estimation) * upper_bounds
        best_arms = np.argwhere(conditions == conditions.max()).reshape(-1)
        return np.random.choice(best_arms)

    def update(self, pulled_arm, outcome, next_purchases, cost):
        self.t += 1
        # compute the mean in an incremental way
        n_samples = len(self.outcome_per_arm[pulled_arm])
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * n_samples + outcome) / (n_samples + 1)
        # update all the upper confidence bounds
        for arm in range(self.n_arms):
            n_samples = len(self.outcome_per_arm[arm])
            self.confidences[arm] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        # at the denominator of the confidences there is N_a @ t-1, so append after the computation of the confidence.
        self.update_observations(pulled_arm, outcome, next_purchases, cost)
