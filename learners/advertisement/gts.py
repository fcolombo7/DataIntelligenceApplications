import numpy as np
from scipy.stats import norm

from learners.learner import Learner


class GTS(Learner):

    LEARNER_NAME = "GTS"

    def __init__(self, arms):
        super().__init__(arm_values=arms)
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*1e3
        
        self.ineligibility = np.zeros(self.n_arms)
        self.negative_threshold = 0.2
        self.penalty = 0.8
        
    def pull_arm(self):
        if self.t < 10:
            return np.random.choice(self.n_arms)
        penalized_means = self.means
        penalized_means[self.ineligibility > self.negative_threshold] = \
            penalized_means[self.ineligibility > self.negative_threshold] * self.penalty
        sampled_values = np.random.normal(penalized_means, self.sigmas)    
        idx = np.argmax(sampled_values)
        return idx
    
    def update(self, pulled_arm, rewards, cost=-1):
        self.t += 1
        reward = rewards['n_clicks'] * (rewards['conv_rates'] * rewards['margin'] * (1 + rewards['tau']) - rewards['cpc'])

        self.outcome_per_arm[pulled_arm].append(reward)
        self.daily_collected_rewards = np.append(self.daily_collected_rewards, reward)

        for arm in range(self.n_arms):
            self.means[arm] = np.mean(self.outcome_per_arm[arm])

        self.ineligibility[pulled_arm] = norm.cdf(0, self.means[pulled_arm], self.sigmas[pulled_arm])
        n_samples = len(self.outcome_per_arm[pulled_arm])
        if n_samples > 1:
            for arm in range(self.n_arms):
                self.sigmas[arm] = np.std(self.outcome_per_arm[arm]) / n_samples
