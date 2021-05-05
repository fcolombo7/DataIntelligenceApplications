import numpy as np


class SimplePricingEnvironment:
    def __init__(self, n_arms, conversion_rates, cost_per_click, n_clicks, margins, tau):
        self.n_arms = n_arms
        self.conversion_rates = conversion_rates
        self.cpc = cost_per_click
        self.n_clicks = n_clicks
        self.margins = margins
        #self.tau = tau

        self.tau = np.zeros(n_arms)
        self.cpc = 0
        self._cost_per_round = self.n_clicks * self.cpc
        self._arm_values = self.margins * (1.0 + self.tau) * self.n_clicks

        self.__compute_expected_rewards()

    def round(self, pulled_arm):
        outcome = np.random.binomial(1, self.conversion_rates[pulled_arm])
        norm_value = (self._arm_values[pulled_arm] - self._cost_per_round) / (np.max(self._arm_values) - self._cost_per_round)
        rew = outcome * norm_value
        actual_rew = outcome * self._arm_values[pulled_arm] - self._cost_per_round
        return rew, actual_rew

    def get_expected_rewards(self):
        return self.expected_rewards

    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            exp = (self.conversion_rates[arm] * self.margins[arm] * (1 + self.tau[arm]) - self.cpc) * self.n_clicks
            self.expected_rewards = np.append(self.expected_rewards, exp)
