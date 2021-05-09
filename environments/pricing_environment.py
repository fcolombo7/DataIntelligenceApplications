import numpy as np


class PricingEnvironment:
    def __init__(self, n_arms, conversion_rates, cost_per_click, n_clicks, tau):
        self.n_arms = n_arms
        self.conversion_rates = conversion_rates
        self.cpc = cost_per_click
        self.n_clicks = n_clicks
        self.tau = tau
        self.future_purchases = {}
        self.selected_arms = {}
        self.t = 0

    def round(self, pulled_arm):
        outcome = np.random.binomial(1, self.conversion_rates[pulled_arm])
        if outcome != 0:
            p = self.tau[pulled_arm]/30
            single_future_purchases = np.random.binomial(30, p)
            self.future_purchases[self.t].append(single_future_purchases)
        return [outcome, self.cpc]

    def day_round(self, pulled_arm):
        self.future_purchases[self.t] = []
        self.selected_arms[self.t] = pulled_arm
        daily_rew = None
        for _ in range(self.n_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm)))
        self.t += 1
        return daily_rew

    def get_future_purchases(self, day):
        if day < 30:
            return None, []
        return self.selected_arms.pop(day - 30), self.future_purchases.pop(day - 30)
