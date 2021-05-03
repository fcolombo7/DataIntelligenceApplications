from learners.Learner import Learner
import numpy as np


class NormalGammaTS(Learner):

    LEARNER_NAME = 'NormalGammaTS'

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.n = np.zeros(n_arms)  # the number of times the arms have been tried

        # TODO: WHY THIS INITIALIZATION???
        # self.alpha = np.array([1] * self.n_arms)  # gamma shape parameter
        # self.beta = np.array([10] * self.n_arms)  # gamma rate parameter
        self.alpha = np.zeros(n_arms)
        self.beta = np.ones(n_arms)

        # self.mu_0 = np.array([1] * self.n_arms)  # the prior (estimated) mean
        # self.v_0 = self.beta / (self.alpha + 1)  # the prior (estimated) variance
        self.mu_0 = np.zeros(n_arms)
        self.v_0 = np.zeros(n_arms)
        self.estimated_variance = None

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        precision = np.random.gamma(self.alpha, 1.0/self.beta)
        for arm in range(self.n_arms):
            if precision[arm] == 0 or self.n[arm] == 0:
                precision[arm] = 0.001
        estimated_variance = 1.0/precision

        self.estimated_variance = estimated_variance
        samples = np.random.normal(self.mu_0, np.sqrt(estimated_variance))
        ids = np.argmax(samples).reshape(-1)
        # print(f't={self.t}: {ids=}, {samples=}')
        return np.random.choice(ids)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        n = 1
        v = self.n[pulled_arm]

        self.alpha[pulled_arm] = self.alpha[pulled_arm] + n / 2
        self.beta[pulled_arm] = self.beta[pulled_arm] + ((n * v / (v + n)) * (((reward - self.mu_0[pulled_arm]) ** 2) / 2))

        # estimate the variance - calculate the mean from the gamma hyper-parameters
        self.v_0[pulled_arm] = self.beta[pulled_arm] / (self.alpha[pulled_arm] + 1)

        self.n[pulled_arm] += 1
        self.mu_0[pulled_arm] = np.array(self.rewards_per_arm[pulled_arm]).mean()


