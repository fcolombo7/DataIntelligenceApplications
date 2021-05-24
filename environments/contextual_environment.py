import numpy as np

from environments.Environment import Environment


class ContextualEnvironment(Environment):

    def __init__(self, bid_idx=3, mode='all', src='src/basic003.json', generator='basic'):
        super().__init__(mode=mode, src=src, generator=generator, bid=bid_idx)
        self.customer_distributions = self.data_gen.get_class_distributions(bid_idx)
        self.selected_bid = bid_idx
        # aggregated or fixed data
        self.class_clicks = self.data_gen.get_daily_clicks()
        self.daily_clicks = np.sum([np.rint(x).astype(int) for x in self.class_clicks[:, bid_idx]])

        self.cpc = self.data_gen.get_costs_per_click(mode='aggregate', bid=bid_idx)
        self.cost_per_click = self.cpc[bid_idx]
        # auxiliary variable
        self.daily_users_categories = None
        self.daily_users_features = None
        self.cur_category_id = None
        self.collected_future_purchases = {}
        self.collected_users_features = {}
        self.selected_arms = {}
        self.day = 0

    # todo: at first call get_user_features, then round [this is due to the cur.user update]
    def round(self, pulled_arm):
        """
        :param pulled_arm:
        :return:
        """
        # determine the outcome of the experiment: buy / not but
        outcome = np.random.binomial(1, self.conv_rates[self.cur_category_id][pulled_arm])
        if outcome != 0:
            p = self.tau[self.cur_category_id][pulled_arm] / 30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        return [outcome, self.cost_per_click]

    def day_round(self, pulled_arms):
        """
        Method that returns the daily rewards according to the arms that have been pulled
        :param pulled_arms: list of tuples composed by a dictionary representing a subset of the feature and
        the arm to pull for users that are in that subset or integer if no context is used
        :return:
        """
        self.sample_daily_users()
        self.collected_future_purchases[self.day + 30] = []
        self.selected_arms[self.day] = []
        daily_rew = None
        for i in range(self.daily_clicks):
            # get the actual user category
            cur_user_category = self.daily_users_categories[i]
            # get the current user features and store them to future use (when the next purchases are retrieved)
            cur_user_features = self.get_daily_user_features(i)
            self.daily_users_features.append(cur_user_features)
            # cur_user_category is the index 'C1'
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
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm)))
            self.selected_arms[self.day].append(pulled_arm)

        # save all the features of all the users
        self.collected_users_features[self.day] = self.daily_users_features
        # increment the day counter
        self.day += 1
        # set the default daily categories/features
        self.daily_users_categories = None
        self.daily_users_features = None
        self.cur_category_id = None

        return daily_rew

    def get_next_purchases_at_day(self, day, keep=True):
        if day not in self.collected_future_purchases.keys():
            return None
        return self.collected_future_purchases[day] if keep else self.collected_future_purchases.pop(day)

    def get_selected_arms_at_day(self, day, keep=True):
        if day not in self.selected_arms.keys():
            return None
        return self.selected_arms[day] if keep else self.selected_arms.pop(day)

    def get_collected_user_features_at_day(self, day, keep=True):
        if day not in self.collected_users_features.keys():
            return None
        return self.collected_users_features[day] if keep else self.collected_users_features.pop(day)

    def sample_daily_users(self, mode='fixed'):
        self.daily_users_features = []
        if mode == 'random':
            self.daily_users_categories = np.random.choice(list(self.customer_classes.keys()),
                                                           size=self.daily_clicks,
                                                           p=self.customer_distributions).tolist()
        elif mode == 'fixed':
            tmp = []
            for i, k in enumerate(self.customer_classes.keys()):
                tmp += [k] * np.rint(self.class_clicks[i, self.selected_bid]).astype(int)
            self.daily_users_categories = np.array(tmp)
            np.random.shuffle(self.daily_users_categories)

        else:
            raise NotImplementedError

        # DEBUG
        # print(f'#### DEBUG ENV (day: {self.day}) ####')
        # counters = [0 for k in self.customer_classes.keys()]
        # for val in self.daily_users_categories:
        #     counters[list(self.customer_classes.keys()).index(val)] += 1
        # for i, k in enumerate(self.customer_classes.keys()):
        #     print(f'{k}: {counters[i]} [{counters[i]/self.daily_clicks}];')
        # print(f'{self.daily_users_categories[:10]} ... (first 10 only)')
        # print('\n')

    def get_daily_user_features(self, index):
        # list of possible combination of the feature space
        cur_user_category = self.daily_users_categories[index]
        cur_user_features = self.customer_classes[cur_user_category]['features']
        idx = np.random.choice(len(cur_user_features))
        return cur_user_features[idx]
