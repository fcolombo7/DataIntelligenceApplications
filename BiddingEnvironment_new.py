import numpy as np
from data_generators.basic_generator import *

def fun(x):
    return 100*(1.0 - np.exp(-4*x + 3*x**3))

class BiddingEnvironment():
    
    def __init__(self, bids, sigma, n_clicks, cpc, conversion_rates, margins, tau, price_idx):
        data_gen = BasicDataGenerator("src/basic002.json")
        
        self.n_arms = len(bids)
        self.bids = bids
        self.n_clicks = n_clicks
        self.cpc = cpc
        self.conv_rates = conversion_rates
        self.margins = margins
        self.tau = tau
        self.price_idx = price_idx
        self.sigmas = np.ones(len(bids))*sigma

        self.__compute_expected_rewards()
        
        
         
            
    def round(self, pulled_arm):
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], self.sigmas[pulled_arm])
        sample_cpc = np.random.normal(self.cpc[pulled_arm], self.sigmas[pulled_arm])
           

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

    '''
   
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        
    def round(self, pulled_arm):
        rewards = np.array([])
        sample_n_clicks = np.random.normal(self.n_clicks[pulled_arm], self.sigmas[pulled_arm])
        sample_cpc = np.random.normal(self.n_clicks[pulled_arm], self.sigmas[pulled_arm])
        reward = sample_n_clicks * (self.conv_rate[self.price_idx] * self.margins[self.price_idx] * \
                    (1 + self.tau) - sample_cpc)
                                    
                                    
        for i in range(0, self.n_clicks):
            rew = np.random.binomial(1, self.conversion_rates[pulled_arm]) * \
                  self.margins[pulled_arm] * (1 + np.random.normal(self.tau[pulled_arm], 1)) - self.cpc
            rewards = np.append(rewards, rew)
        return rewards

    def day_round(self, pulled_arm):
        return np.sum(self.round(pulled_arm))

    def get_opt(self):
        return np.max(self.expected_rewards)

    def __compute_expected_rewards(self):
        self.expected_rewards = np.array([])
        for arm in range(0, self.n_arms):
            exp = self.conversion_rates[arm] * self.margins[arm] * (1 + self.tau[arm]) - self.cpc
            self.expected_rewards = np.append(self.expected_rewards, exp)
    '''

