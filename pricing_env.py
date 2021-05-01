import numpy as np


class PricingEnvironment:
    def __init__(self, n_arms, conversion_rate, cpc, n_clicks, margin, tau):
        self.n_arms = n_arms
        self.conversion_rate = conversion_rate
        self.cpc = cpc
        self.n_clicks = n_clicks
        self.margin = margin
        self.tau = tau

    def round(self, pulled_arm):
        rewards = np.array([])
        for i in range(int(np.floor(self.n_clicks))):
            rewards = np.append(rewards, (np.random.binomial(1, self.conversion_rate[pulled_arm]) * self.margin[pulled_arm] * (1 + np.random.normal(self.tau[pulled_arm], 1)) - self.cpc))
        
        return rewards

