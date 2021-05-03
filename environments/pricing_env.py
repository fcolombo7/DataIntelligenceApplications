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
        self.norm_max = np.max(self.margins)*(1 + np.max(self.tau) + 3*1)  # 3 * dev standard => 99.97 %
        self.norm_min = 0.0 - self.cpc

    def round(self, pulled_arm):
        rewards = np.array([])
        for i in range(0, self.n_clicks):
            rew = np.random.binomial(1, self.conversion_rates[pulled_arm]) * \
                  self.margins[pulled_arm] * (1 + np.random.normal(self.tau[pulled_arm], 1)) - self.cpc
            rew = (rew - self.norm_min) / (self.norm_max - self.norm_min)
            rewards = np.append(rewards, rew)
        return rewards

    def day_round(self, pulled_arm):  # TODO: questo mi sa che è una cazzata
        return np.sum(self.round(pulled_arm))

    def get_expected_rewards(self, normalize=True):
        if normalize:
            return self.conversion_rates * self.norm_expected_rewards
        return self.expected_rewards

    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        self.norm_expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            exp = self.conversion_rates[arm] * self.margins[arm] * (1 + self.tau[arm]) - self.cpc
            self.expected_rewards = np.append(self.expected_rewards, exp)
            # normalized
            norm_exp = self.margins[arm] * (1 + self.tau[arm]) - self.cpc
            self.norm_expected_rewards = np.append(self.norm_expected_rewards, norm_exp)
        self.norm_expected_rewards = (self.norm_expected_rewards - np.min(self.norm_expected_rewards)) / \
                                     (np.max(self.norm_expected_rewards) - np.min(self.norm_expected_rewards))
