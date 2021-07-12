import numpy as np

from data_generators.standard_generator import StandardDataGenerator


class CompleteEnvironment:
    def __init__(self, src='src/basic003.json'):
        self.data_gen = StandardDataGenerator(src)

        self.bids = self.data_gen.get_bids()
        self.prices = self.data_gen.get_prices()
        self.margins = self.data_gen.get_margins()
        self.conv_rates = self.data_gen.get_conversion_rates()
        self.class_clicks = self.data_gen.get_daily_clicks()
        self.class_costs_per_click = self.data_gen.get_costs_per_click()
        self.class_future_purchases = self.data_gen.get_future_purchases()
        self.features = self.data_gen.get_features()
        self.customer_classes = self.data_gen.get_classes()

        # auxiliary variable
        self.daily_users_categories = None
        self.cur_category_id = None
        self.selected_features = None
        self.collected_future_purchases = {}
        self.collected_users_features = {}
        self.selected_arms = {}
        self.day = 0

        # save the current round params
        self.sampled_n_clicks = None
        self.sampled_cpc = None

    def day_round(self, pulled_arms, selected_bid, fixed_adv=False):
        """
        Method that returns the daily rewards according to the arms that have been pulled
        :param pulled_arms: list of tuples composed by a dictionary representing a subset of the feature and
        the arm to pull for users that are in that subset or integer if no context is used
        :param selected_bid:
        :param fixed_adv: simulation of the adv campaign. If true, the number of clicks and the cost per clicks are
        fixed, otherwise they are sampled.
        :return:
        """
        self._sample_bidding_params(selected_bid, fixed_adv)
        self._sample_daily_users()
        self.collected_future_purchases[self.day + 30] = []
        self.selected_arms[self.day] = []
        self.collected_users_features[self.day] = []

        daily_clicks = len(self.daily_users_categories)
        daily_rew = []
        for i in range(daily_clicks):
            # get the actual user category
            cur_user_category = self.daily_users_categories[i]
            # get the current user features and store them to future use (when the next purchases are retrieved)
            cur_user_features = self.get_daily_user_features(i)
            # cur_user_category is the index which is provided in the json source [so it is like 'C1'].
            self.cur_category_id = list(self.customer_classes.keys()).index(cur_user_category)
            # now determine which is the arm to pull according to the context received by the learner
            if type(pulled_arms) is not list:
                pulled_arm = pulled_arms
            else:
                pulled_arm = None
                for context, arm in pulled_arms:
                    good_context = True
                    for feature in context:
                        feature_id = self.features.index(feature)
                        if cur_user_features[feature_id] != context[feature]:
                            good_context = False
                            break
                    if good_context:
                        pulled_arm = arm
                        break
            daily_rew.append(self._round(pulled_arm))
            self.selected_arms[self.day].append(pulled_arm)
            self.collected_users_features[self.day].append(cur_user_features)

        # increment the day counter
        self.day += 1
        # set the default daily categories/features
        self.daily_users_categories = None
        # self.daily_users_features = None
        self.cur_category_id = None
        return daily_rew, sum(self.sampled_n_clicks), np.average(self.sampled_cpc, weights=self.sampled_n_clicks)

    def get_next_purchases_at_day(self, day, keep=True, filter_purchases=True):
        if day not in self.collected_future_purchases.keys():
            return None
        if not filter_purchases:
            return self.collected_future_purchases[day] if keep else self.collected_future_purchases.pop(day)
        purch = self.collected_future_purchases[day] if keep else self.collected_future_purchases.pop(day)
        mask = ~np.isnan(np.array(purch).astype(float))
        return list(np.array(purch)[mask])

    def get_selected_arms_at_day(self, day, keep=True, filter_purchases=False):
        if day not in self.selected_arms.keys():
            return None
        if not filter_purchases:
            return self.selected_arms[day] if keep else self.selected_arms.pop(day)
        purch = self.get_next_purchases_at_day(day + 30, filter_purchases=False)
        if purch is None:
            return None
        mask = ~np.isnan(np.array(purch).astype(float))
        arms = self.selected_arms[day] if keep else self.selected_arms.pop(day)
        return list(np.array(arms)[mask])

    def get_collected_user_features_at_day(self, day, keep=True, filter_purchases=False):
        if day not in self.collected_users_features.keys():
            return None
        if not filter_purchases:
            return self.collected_users_features[day] if keep else self.collected_users_features.pop(day)
        purch = self.get_next_purchases_at_day(day + 30, filter_purchases=False)
        if purch is None:
            return None
        mask = ~np.isnan(np.array(purch).astype(float))
        features = self.collected_users_features[day] if keep else self.collected_users_features.pop(day)
        return list(np.array(features)[mask])

    def get_daily_user_features(self, index):
        # list of possible combination of the feature space
        cur_user_category = self.daily_users_categories[index]
        cur_user_features = self.customer_classes[cur_user_category]['features']
        idx = np.random.choice(len(cur_user_features))
        return cur_user_features[idx]

    def _sample_daily_users(self):
        tmp = []
        for i, k in enumerate(self.customer_classes.keys()):
            tmp += [k] * self.sampled_n_clicks[i].astype(int)
        self.daily_users_categories = np.array(tmp)
        np.random.shuffle(self.daily_users_categories)
        self.daily_users_categories = self.daily_users_categories.tolist()

    # todo: at first call get_user_features, then round [this is due to the cur.user update]
    def _round(self, pulled_arm):
        """
        :param pulled_arm: selected price
        :param
        :return:
        """
        # determine the outcome of the experiment: buy / not but
        outcome = np.random.binomial(1, self.conv_rates[self.cur_category_id][pulled_arm])
        single_future_purchases = None
        if outcome != 0:
            p = self.class_future_purchases[self.cur_category_id][pulled_arm] / 30
            single_future_purchases = np.random.binomial(30, p)
        self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        return outcome, self.sampled_cpc[self.cur_category_id]

    def _sample_bidding_params(self, selected_bid, fixed_adv):
        if not fixed_adv:
            self.sampled_n_clicks = np.around(np.random.normal(self.class_clicks[:, selected_bid],
                                                               abs(self.class_clicks[:, selected_bid] / 100)))
            self.sampled_cpc = np.around(np.random.normal(self.class_costs_per_click[:, selected_bid],
                                                          abs(self.class_costs_per_click[:, selected_bid] / 100)),
                                         decimals=3)
        else:
            self.sampled_n_clicks = np.around(self.class_clicks[:, selected_bid])
            self.sampled_cpc = np.around(self.class_costs_per_click[:, selected_bid], decimals=3)
