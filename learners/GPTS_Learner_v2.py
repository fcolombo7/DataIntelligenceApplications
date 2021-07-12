from learners.pricing.learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm


class GPTS_Learner(Learner):
    LEARNER_NAME = "GPTS"

    def __init__(self, arms):
        super().__init__(arm_values=arms)
        self.arms = arms
        self.day = 0
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.ineligibility = np.ones(self.n_arms)
        self.negative_threshold = 0.2
        self.pulled_arms = []

        alpha = 40.0
        kernel = C(1.0, (1e-2, 1e5)) * RBF(1.0, (1e-2, 1e5))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, arm_idx, rewards, cost):
        reward = rewards['n_clicks'] * (
                    rewards['conv_rates'] * rewards['margin'] * (1 + rewards['tau']) - rewards['cpc'])
        super().update_observations(arm_idx, reward, cost)
        self.ineligibility = norm.cdf(0, self.means, self.sigmas)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.daily_collected_rewards

        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, rewards, cost=-1):
        self.update_observations(pulled_arm, rewards, cost)
        self.next_day()
        self.update_model()

    def pull_arm(self):
        if self.day < 10:
            arm = np.random.choice(self.n_arms)
            return arm
        sampled_values = np.random.normal(self.means, self.sigmas)
        sampled_values[self.ineligibility > self.negative_threshold] = 0
        return np.argmax(sampled_values)
