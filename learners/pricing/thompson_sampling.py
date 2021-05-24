from learners.pricing.learner import Learner
import numpy as np


class ThompsonSampling(Learner):

    LEARNER_NAME = 'TS-Pricing'

    def __init__(self, arm_values, period=30, next_purchases_update='binomial'):
        super().__init__(arm_values, period, next_purchases_update)
        self.beta_parameters = np.ones((self.n_arms, 2))

    def __str__(self):
        s = f'learner_type={self.LEARNER_NAME}\n'
        s += super(ThompsonSampling, self).__str__()
        opt = self.get_opt_arm_expected_value()
        s += f'opt_arm_value: {opt[0]} [{opt[1]}]'
        return s

    def pull_arm(self) -> int:
        # TODO: mean, not parameter of the binomial
        #  factor = self.arm_values * (1 + self.period * self.next_purchases_estimation)
        factor = self.arm_values * (1 + self.next_purchases_estimation)
        ids = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * factor).reshape(-1)
        return np.random.choice(ids)

    def update(self, pulled_arm, outcome, cost):
        self.t += 1
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + outcome
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - outcome
        self.update_observations(pulled_arm, outcome, cost)

    def get_opt_arm_expected_value(self) -> (float, int):
        mean_values = self.beta_parameters[:, 0] / (self.beta_parameters[:, 0] + self.beta_parameters[:, 1])
        factor = self.arm_values * (1 + self.next_purchases_estimation)
        return np.max(mean_values * factor), np.argmax(mean_values * factor)
