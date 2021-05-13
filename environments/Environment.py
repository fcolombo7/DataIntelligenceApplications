from abc import ABC, abstractmethod
from data_generators.basic_generator import *
from sklearn import preprocessing


class Environment(ABC):
    """Environment abstract base class.
    Constructor method uploads all the basic data from the given json source in the given mode"""

    def __init__(self, mode='all', src='src/basic003.json'):
        self.data_gen = BasicDataGenerator(src)

        self.bids = self.data_gen.get_bids()
        self.prices = self.data_gen.get_prices()
        self.margins = self.data_gen.get_margins()
        self.n_clicks = self.data_gen.get_daily_clicks(mode=mode)
        self.cpc = self.data_gen.get_costs_per_click(mode=mode)
        self.conv_rates = self.data_gen.get_conversion_rates(mode=mode)
        self.tau = self.data_gen.get_future_purchases(mode=mode)

    @abstractmethod
    def round(self, pulled_arm):
        """Play a single round of the environment"""
        pass
