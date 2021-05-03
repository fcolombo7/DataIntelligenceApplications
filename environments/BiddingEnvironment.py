import numpy as np
from data_generators.basic_generator import *

def fun(x):
    return 100*(1.0 - np.exp(-4*x + 3*x**3))

class BiddingEnvironment():
    
    def __init__(self, bids, sigma):
        data_gen = BasicDataGenerator("src/basic002.json")
        self.bids = bids
        self.means = fun(bids) #data_gen.get_daily_clicks(mode = 'aggregate')
        self.sigmas = np.ones(len(bids))*sigma
        
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        


