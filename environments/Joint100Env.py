import numpy as np
from environments.Environment import Environment


class JointEnvironment(Environment):
    def __init__(self, mode='all', src='src/basic003.json'):
        super().__init__(mode=mode, src=src)
        self.prices_arms = len(self.prices)
        self.bids_arms = len(self.bids)
        self.collected_future_purchases = {}
        self.selected_arms = {}
        self.day = 0

        print(f"Environment created with no fixed arm values")

    def round(self, pulled_arm):
        outcome = np.random.binomial(1, self.conv_rates[pulled_arm//self.prices_arms])
        if outcome != 0:
            p = self.tau[pulled_arm//self.prices_arms]/30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        sample_cpc = np.random.normal(self.cpc[pulled_arm % self.bids_arms], abs(self.cpc[pulled_arm % self.bids_arms] / 100))
        return [outcome, sample_cpc]

    def day_round(self, pulled_arm):
        self.collected_future_purchases[self.day + 30] = []
        self.selected_arms[self.day] = []
        daily_rew = None
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm % self.bids_arms], abs(self.n_clicks[pulled_arm % self.bids_arms] / 100))
        n_clicks = np.rint(sample_n_clicks).astype(int)
        for _ in range(n_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm//self.prices_arms)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm//self.prices_arms)))
            self.selected_arms[self.day].append(pulled_arm//self.prices_arms)
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