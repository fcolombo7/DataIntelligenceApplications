from environments.Environment import *
from sklearn import preprocessing


class BiddingEnvironment(Environment):
    
    def __init__(self, bids, sigma, price_idx=4, mode='all', src='src/basic003.json'):
        super().__init__(mode=mode, src=src)
        self.n_arms = len(self.bids)
        self.price_idx = price_idx
        self.sigmas = np.ones(len(self.bids))*sigma
        
        self.min_rew = -500
        self.max_rew = 1000
       
        self.__compute_expected_rewards()
        
        print(f"Environment created with fixed price: {self.prices[price_idx]}")
       
    def round(self, pulled_arm):
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], abs(self.n_clicks[pulled_arm]/100))
        sample_cpc = np.random.normal(self.cpc[pulled_arm], abs(self.cpc[pulled_arm]/100))

        reward = sample_n_clicks * (self.conv_rates[self.price_idx] * self.margins[self.price_idx] * \
                    (1 + self.tau[self.price_idx]) - sample_cpc)
        return reward
    
    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            exp = self.n_clicks[arm] * (self.conv_rates[self.price_idx] * self.margins[self.price_idx] * \
                       (1 + self.tau[self.price_idx]) - self.cpc[arm])
            norm_exp = (exp - self.min_rew)/(self.max_rew - self.min_rew)            
            self.expected_rewards = np.append(self.expected_rewards, exp)
            
    def get_opt(self):
        return np.max(self.expected_rewards)
        
    def get_opt_arm(self):
        return np.argmax(self.expected_rewards)
        
    def expected_rew(self):
        return self.expected_rewards
    