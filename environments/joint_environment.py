import numpy as np
from environments.Environment import Environment


class JointEnvironment(Environment):
    def __init__(self, bid_idx=3, mode='all', src='src/basic003.json'):
        super().__init__(mode=mode, src=src)
        self.bid_idx = bid_idx
#        self.n_clicks = np.rint(self.n_clicks[bid_idx]).astype(int)
#        self.cpc = self.cpc[bid_idx]
        self.collected_future_purchases = {}
        self.selected_arms = {}
        self.day = 0

        print(f"Environment created with fixed bid: {self.bids[bid_idx]}")

    def round(self, pulled_arm):
        used_cpc = self.cpc[self.bid_idx]
        outcome = np.random.binomial(1, self.conv_rates[pulled_arm])
        if outcome != 0:
            p = self.tau[pulled_arm]/30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        return [outcome, used_cpc]

    def day_round(self, pulled_arm):
        self.collected_future_purchases[self.day + 30] = []
        self.selected_arms[self.day] = []
        daily_rew = None
        n_clicks = np.rint(self.n_clicks[self.bid_idx]).astype(int)
        for _ in range(n_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm)))
            self.selected_arms[self.day].append(pulled_arm)
        self.day += 1
        return daily_rew
    
    def bidding_round(self, pulled_arm):
        rewards = {}
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], abs(self.n_clicks[pulled_arm]/100))
        sample_cpc = np.random.normal(self.cpc[pulled_arm], abs(self.cpc[pulled_arm]/100))

        rewards['n_clicks'] = sample_n_clicks
        rewards['cpc'] = sample_cpc
        rewards['margin'] = self.margins
        rewards['tau'] = self.tau
        rewards['conv_rates'] = self.conv_rates
        
        return rewards 

    def get_next_purchases_at_day(self, day, keep=True):
        if day not in self.collected_future_purchases.keys():
            return None
        return self.collected_future_purchases[day] if keep else self.collected_future_purchases.pop(day)

    def get_selected_arms_at_day(self, day, keep=True):
        if day not in self.selected_arms.keys():
            return None
        return self.selected_arms[day] if keep else self.selected_arms.pop(day)
