import numpy as np

from learners.GPTS_Learner_v2 import GPTS_Learner
from learners.pricing.thompson_sampling import ThompsonSampling
from scipy.stats import norm


class JointLearner:

    LEARNER_NAME = 'JointBandit'

    def __init__(self,
                 margin_values,
                 bid_values,
                 pricing_bandit_class=ThompsonSampling,
                 adv_bandit_class=GPTS_Learner):
        self.margins = margin_values
        self.bids = bid_values
        self.pricing_bandit = pricing_bandit_class(self.margins)
        self.adv_bandit = adv_bandit_class(self.bids)
        self.sel_bid = None
        self.sel_price = None
        self.discovery = True

    def pull_arm(self):
        """
        This method returns the pair bid - price to pull.
        :return:
        """
        self.sel_bid = self.adv_bandit.pull_arm()
        self.sel_price = self.pricing_bandit.pull_arm()
        return self.sel_bid, self.sel_price

    def update(self, pulled_price, outcome, cost):
        self.pricing_bandit.update(pulled_price, outcome, cost)

    def get_adv_params(self, n_clicks, cpc):
        self.n_clicks = n_clicks
        self.cpc = cpc

    def next_day(self):
        self.pricing_bandit.next_day()
        if self.pricing_bandit.day > 0:
            params = {
                'n_clicks': self.n_clicks,
                'cpc': self.cpc,
                'tau': self.pricing_bandit.next_purchases_estimation[self.sel_price],
                'margin': self.margins[self.sel_price],
                'conv_rates': self.pricing_bandit.get_estimated_conv_rates()[self.sel_price] #TODO: DA SISTEAMRE ANCHE I PUNTI PRECEDENTI
            }
            self.adv_bandit.update(self.sel_bid, params)

    # TODO: check all this methods
    # other methods to ensure compatibility with the simulations
    def get_next_purchases_data(self):
        self.pricing_bandit.get_next_purchases_data()

    def set_next_purchases_data(self, estimation, observations, update_mode):
        self.pricing_bandit.set_next_purchases_data(estimation, observations, update_mode)

    def update_single_future_purchase(self, price_arm, single_obs):
        if self.discovery:
            self.discovery = False
            self.adv_bandit.sigmas = np.ones(self.adv_bandit.n_arms) * 10
            self.adv_bandit.pulled_arms = []
            self.adv_bandit.daily_collected_rewards = np.array([])
            self.pricing_bandit.beta_parameters = np.ones((self.pricing_bandit.n_arms, 2))
        self.pricing_bandit.update_single_future_purchase(price_arm, single_obs)

    def get_collected_reward(self):
        return self.pricing_bandit.daily_collected_rewards
