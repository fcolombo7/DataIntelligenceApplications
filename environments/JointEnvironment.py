from environments.Environment import *
from sklearn import preprocessing


class JointEnvironment(Environment):
    
    def __init__(self, bids, prices, sigma, mode='all', src='src/basic003.json'):
        super().__init__(mode=mode, src=src)
        self.n_arms = len(self.bids)
        self.sigmas = np.ones(len(self.bids))*sigma
        self.bids = bids
        self.prices = prices
        
        self.__compute_expected_rewards()
       
    def round(self, pulled_arm):
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], abs(self.n_clicks[pulled_arm]/100))
        sample_cpc = np.random.normal(self.cpc[pulled_arm], abs(self.cpc[pulled_arm]/100))
        
        return sample_n_clicks, sample_cpc
    
    def pricing_day_round(self, pulled_arm, n_clicks, cpc):
        self.collected_future_purchases[self.day + 30] = []
        self.selected_arms[self.day] = []
        daily_rew = None
        for _ in range(0, n_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm, cpc)))
            self.selected_arms[self.day].append(pulled_arm)
        self.day += 1
        return daily_rew
    
    def pricing_round(self, pulled_arm, cpc):
        outcome = np.random.binomial(1, self.conversion_rates[pulled_arm])
        if outcome != 0:
            p = self.tau[pulled_arm]/30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        return [outcome, cpc]    
        
    
    

    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for bid_idx in range(0, len(self.bids)):
            for price_idx in range(0, len(self.prices)):
                exp = self.n_clicks[bid_idx] * (self.conv_rates[price_idx] * self.margins[price_idx] * \
                       (1 + self.tau[price_idx]) - self.cpc[bid_idx])
                self.expected_rewards = np.append(self.expected_rewards, exp)
            
    def get_opt(self):
        return np.max(self.expected_rewards)
        
    def get_opt_arm(self):
        return np.argmax(self.expected_rewards)
        
    def expected_rew(self):
        return self.expected_rewards
    