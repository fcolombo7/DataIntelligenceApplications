import numpy as np


class PricingEnvironment:
    def __init__(self, conversion_rates, cost_per_click, n_clicks, tau):
        self.conversion_rates = conversion_rates
        self.cpc = cost_per_click
        self.n_clicks = n_clicks
        self.tau = tau

        self.collected_future_purchases = {}
        self.selected_arms = {}
        self.day = 0

    def round(self, pulled_arm):
        outcome = np.random.binomial(1, self.conversion_rates[pulled_arm])
        if outcome != 0:
            p = self.tau[pulled_arm]/30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        return [outcome, self.cpc]

    def day_round(self, pulled_arm):
        self.collected_future_purchases[self.day + 30] = []
        self.selected_arms[self.day] = []
        daily_rew = None
        for _ in range(self.n_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm)))
            self.selected_arms[self.day].append(pulled_arm)
        self.day += 1
        return daily_rew

    def get_next_purchases_at_day(self, day, keep=True):
        if day not in self.collected_future_purchases.keys():
            return None
        return self.collected_future_purchases[day] if keep else self.collected_future_purchases.pop(day)

    def get_selected_arms_at_day(self, day, keep=True):
        if day not in self.selected_arms.keys():
            return None
        return self.selected_arms[day] if keep else self.selected_arms.pop(day)
