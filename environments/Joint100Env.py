import numpy as np
from environments.Environment import Environment


class JointEnvironment(Environment):
    def __init__(self, mode='all', src='src/basic003.json', generator='standard'):
        super().__init__(mode=mode, src=src, generator=generator)
        self.prices_arms = len(self.prices)
        self.mode = mode
        self.bids_arms = len(self.bids)
        self.collected_future_purchases = {}
        self.selected_arms = {}
        self.day = 0
        self.__compute_expected_rewards()

        print(f"Environment created with no fixed arm values")

    def round(self, pulled_arm):
        self.__update_parameters(pulled_arm % self.bids_arms)
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

    def __update_parameters(self, pulled_arm):
        self.conv_rates = self.data_gen.get_conversion_rates(mode=self.mode, bid=pulled_arm)
        self.tau = self.data_gen.get_future_purchases(mode=self.mode, bid=pulled_arm)
        self.cpc = self.data_gen.get_costs_per_click(mode=self.mode, bid=pulled_arm)

    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        arms = np.array(np.meshgrid(range(0, self.prices_arms), range(0, self.bids_arms))).T.reshape(-1, 2)
        for bid, price in arms:
            conv_rate = self.data_gen.get_conversion_rates(mode=self.mode, bid=bid)
            tau = self.data_gen.get_future_purchases(mode=self.mode, bid=bid)
            cpc = self.data_gen.get_costs_per_click(mode=self.mode, bid=bid)
            exp = self.n_clicks[bid] * (conv_rate[price] * self.margins[price] * (1 + tau[price]) - cpc[bid])
            self.expected_rewards = np.append(self.expected_rewards, exp)

    def get_opt(self):
        return np.max(self.expected_rewards)

    def get_opt_arm(self):
        return np.argmax(self.expected_rewards)
