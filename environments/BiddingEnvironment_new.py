from environments.Environment import *


class BiddingEnvironment(Environment):
    
    def __init__(self, bids, sigma, price_idx=0, mode='all', src='src/basic002.json'):
        super().__init__(mode=mode, src=src)
        self.n_arms = len(bids)
        self.price_idx = price_idx
        self.sigmas = np.ones(len(bids))*sigma

        self.__compute_expected_rewards()

        print(f"Environment created with fixed price: {self.prices[price_idx]}")

    def round(self, pulled_arm):
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], self.sigmas[pulled_arm])
        sample_cpc = np.random.normal(self.cpc[pulled_arm], self.sigmas[pulled_arm]/10)

        reward = sample_n_clicks * (self.conv_rates[self.price_idx] * self.margins[self.price_idx] * \
                    (1 + self.tau[self.price_idx]) - sample_cpc)
        
        return reward
    
    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            exp = self.n_clicks[arm] * (self.conv_rates[self.price_idx] * self.margins[self.price_idx] * \
                       (1 + self.tau[self.price_idx]) - self.cpc[arm])
            self.expected_rewards = np.append(self.expected_rewards, exp)
            
    def get_opt(self):
        return np.max(self.expected_rewards)