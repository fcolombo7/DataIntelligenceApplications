from environments.environment import *
from sklearn import preprocessing


class BiddingEnvironment(Environment):
    
    def __init__(self, sigma, price_idx=4, mode='all', src='src/basic003.json', generator='standard'):
        super().__init__(mode=mode, src=src, generator=generator)
        self.n_arms = len(self.bids)
        self.price = price_idx
        self.mode = mode
        self.margins = self.margins[price_idx]
        self.sigmas = np.ones(len(self.bids))*sigma
        self.__compute_expected_rewards()
        
        print(f"Environment created with fixed price: {self.prices[price_idx]}")
       
    def round(self, pulled_arm):
        rewards = {}
        self.__update_parameters(pulled_arm)
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], abs(self.n_clicks[pulled_arm]/100))
        sample_cpc = np.random.normal(self.cpc[pulled_arm], abs(self.cpc[pulled_arm]/100))

        rewards['n_clicks'] = sample_n_clicks
        rewards['cpc'] = sample_cpc
        rewards['margin'] = self.margins
        rewards['tau'] = self.tau
        rewards['conv_rates'] = self.conv_rates
        return rewards
    
    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            conv_rate = self.data_gen.get_conversion_rates(mode=self.mode, bid=arm)[self.price]
            tau = self.data_gen.get_future_purchases(mode=self.mode, bid=arm)[self.price]
            cpc = self.data_gen.get_costs_per_click(mode=self.mode, bid=arm)
            exp = self.n_clicks[arm] * (conv_rate * self.margins * (1 + tau) - cpc[arm])
            self.expected_rewards = np.append(self.expected_rewards, exp)

    def __update_parameters(self, pulled_arm):
        self.conv_rates = self.data_gen.get_conversion_rates(mode=self.mode, bid=pulled_arm)[self.price]
        self.tau = self.data_gen.get_future_purchases(mode=self.mode, bid=pulled_arm)[self.price]
        self.cpc = self.data_gen.get_costs_per_click(mode=self.mode, bid=pulled_arm)

    def get_opt(self):
        return np.max(self.expected_rewards)
        
    def get_opt_arm(self):
        return np.argmax(self.expected_rewards)
        
    def expected_rew(self):
        return self.expected_rewards
