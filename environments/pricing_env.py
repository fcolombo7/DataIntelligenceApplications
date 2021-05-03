import numpy as np


class PricingEnvironment:
    def __init__(self, n_arms, conversion_rates, cost_per_click, n_clicks, margins, tau):
        self.n_arms = n_arms
        self.conversion_rates = conversion_rates
        self.cpc = cost_per_click
        self.n_clicks = n_clicks
        self.margins = margins
        self.tau = tau
        self.__compute_expected_rewards()

    def round(self, pulled_arm):
        rewards = np.array([])
        for i in range(0, self.n_clicks):
            rew = np.random.binomial(1, self.conversion_rates[pulled_arm]) * \
                  self.margins[pulled_arm] * (1 + np.random.normal(self.tau[pulled_arm], 1)) - self.cpc
            rewards = np.append(rewards, rew)
        return rewards

    def day_round(self, pulled_arm):
        return np.sum(self.round(pulled_arm))

    def get_opt(self):
        return np.max(self.expected_rewards)

    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            exp = self.conversion_rates[arm] * self.margins[arm] * (1 + self.tau[arm]) - self.cpc
            self.expected_rewards = np.append(self.expected_rewards, exp)
