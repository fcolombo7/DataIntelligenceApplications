from learners.pricing.learner import Learner
import numpy as np


class ThompsonSampling(Learner):

    LEARNER_NAME = 'TS-Pricing'

    def __init__(self, arm_values, period=30, next_purchases_update='binomial'):
        super().__init__(arm_values, period, next_purchases_update)
        self.beta_parameters = np.ones((self.n_arms, 2))

    def pull_arm(self) -> int:
        factor = self.arm_values * (1 + self.period * self.next_purchases_estimation)
        ids = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * factor).reshape(-1)
        return np.random.choice(ids)

    def update(self, pulled_arm, outcome, next_purchases, cost):
        self.t += 1
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + outcome
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - outcome
        self.update_observations(pulled_arm, outcome, next_purchases, cost)



