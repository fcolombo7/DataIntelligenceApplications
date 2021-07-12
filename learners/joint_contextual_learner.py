import numpy as np

from learners.GPTS_Learner_v3 import GPTS_Learner
from learners.pricing.contextual_learner import ContextualLearner
from learners.pricing.thompson_sampling import ThompsonSampling
from scipy.stats import norm


class JointContextualLearner:

    LEARNER_NAME = 'JointContextualBandit'

    def __init__(self,
                 margin_values,
                 bid_values,
                 features,
                 pricing_bandit_class=ThompsonSampling,
                 adv_bandit_class=GPTS_Learner):
        self.margins = margin_values
        self.bids = bid_values
        self.features = features

        self.pricing_bandit_class = pricing_bandit_class
        self.adv_bandit = adv_bandit_class(self.bids)
        bandit_args = {
            'arm_values': self.margins
        }
        self.pricing_contextual_bandit = ContextualLearner(self.features, self.pricing_bandit_class, **bandit_args)

        self.sel_bid = None
        self.sel_prices = None
        self.n_clicks = None
        self.cpc = None
        self.discovery = True

    def pull_arms(self):
        """
        This method returns the pair bid - structure of arm to pull according to the context.
        :return:
        """
        self.sel_bid = self.adv_bandit.pull_arm()
        self.sel_prices = self.pricing_contextual_bandit.pull_arms()
        return self.sel_bid, self.sel_prices

    def get_adv_params(self, n_clicks, cpc):
        self.n_clicks = n_clicks
        self.cpc = cpc

    def update(self, daily_reward, pulled_prices, user_features):
        # TODO: CHECK HERE!!!
        contexts_distribution = self.pricing_contextual_bandit.update(daily_reward, pulled_prices, user_features)
        #print(f"DISTRIB: {contexts_distribution}")
        reward = 0
        leaves = self.pricing_contextual_bandit.context_tree.get_leaves()
        for i, context_click in enumerate(contexts_distribution):
            #print(self.sel_prices[i][1])
            conv_rate = leaves[i].base_learner.get_estimated_conv_rates()[self.sel_prices[i][1]]
            #print(f"C_RATE: {conv_rate}")
            tau = leaves[i].base_learner.next_purchases_estimation[self.sel_prices[i][1]]
            #print(f"TAU: {tau}")
            margin = self.margins[self.sel_prices[i][1]]
            #print(f"MARGIN: {margin}")
            context_reward = context_click * (conv_rate * margin * (1 + tau) - self.cpc)
            reward += context_reward
            #print(f"C_REWARD: {context_reward}")
        #print(f"REWARD: {reward}")

        # MANUAL UPDATE OF THE ADV BANDIT
        self.adv_bandit.daily_collected_rewards = np.append(self.adv_bandit.daily_collected_rewards, reward)
        self.adv_bandit.ineligibility = norm.cdf(0, self.adv_bandit.means, self.adv_bandit.sigmas)
        self.adv_bandit.pulled_arms.append(self.bids[self.sel_bid])
        self.adv_bandit.day += 1
        self.adv_bandit.update_model()

    def next_day(self):
        self.pricing_contextual_bandit.next_day()

    def update_next_purchases(self, pulled_prices_data: list, next_purchases_data: list, features_data: list):
        if self.discovery:
            self.discovery = False
            self.adv_bandit.sigmas = np.ones(self.adv_bandit.n_arms) * 10
            self.adv_bandit.pulled_arms = []
            self.adv_bandit.daily_collected_rewards = np.array([])
            self.pricing_contextual_bandit.context_tree.beta_parameters = np.ones((len(self.margins), 2))
            self.adv_bandit.day = 0
        self.pricing_contextual_bandit.update_next_purchases(pulled_prices_data, next_purchases_data, features_data)

    def get_collected_reward(self):
        return self.pricing_contextual_bandit.get_daily_rewards()

    def update_context_tree(self, new_context_tree):
        self.pricing_contextual_bandit.update_context_tree(new_context_tree)

    def get_root_learner(self):
        return self.pricing_contextual_bandit.get_root_learner()

    def get_splits_count(self):
        return self.pricing_contextual_bandit.get_splits_count()
