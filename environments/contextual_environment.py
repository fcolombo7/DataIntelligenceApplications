import numpy as np


class ContextualEnvironment:

    def __init__(self, features: [], customer_classes: {}, conversion_rates: [], future_purchases: [],
                 daily_clicks: int, cost_per_click: float):
        self.features = features
        self.customer_classes = customer_classes
        self.customer_distributions = []
        for cc in customer_classes.values():
            self.customer_distributions.append(cc['fraction'])
        # per class data
        self.conversion_rates = conversion_rates
        self.future_purchases = future_purchases
        # aggregated or fixed data
        self.daily_clicks = daily_clicks
        self.cost_per_click = cost_per_click
        # auxiliary variable
        self.daily_users_categories = None
        self.daily_users_features = None
        self.cur_user = None
        self.collected_future_purchases = {}
        self.collected_users_features = {}
        self.selected_arms = {}
        self.day = 0

    # todo: at first call get_user_features, then round [this is due to the cur.user update]
    def round(self, pulled_arm, user_category):
        """
        :param pulled_arm:
        :param user_category:
        :return:
        """
        # determine the outcome of the experiment: buy / not but
        outcome = np.random.binomial(1, self.conversion_rates[user_category][pulled_arm])
        if outcome != 0:
            p = self.future_purchases[user_category][pulled_arm] / 30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.day + 30].append(single_future_purchases)
        return [outcome, self.cost_per_click]

    def day_round(self, pulled_arms):
        """
        Method that returns the daily rewards according to the arms that have been pulled
        :param pulled_arms: list of tuples composed by a dictionary representing a subset of the feature and
        the arm to pull for users that are in that subset.
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
            cur_user_category_id = list(self.customer_classes.keys()).index(cur_user_category)
            # now determine which is the arm to pull according to the context received by the learner
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
                daily_rew = self.round(pulled_arm, cur_user_category_id)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm, cur_user_category_id)))
            self.selected_arms[self.day].append(pulled_arm)

        # save all the features of all the users
        self.collected_users_features[self.day] = self.daily_users_features
        # increment the day counter
        self.day += 1
        # set the default daily categories/features
        self.daily_users_categories = None
        self.daily_users_features = None
        self.cur_user = None

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

    def sample_daily_users(self):
        self.cur_user = 0
        self.daily_users_features = []
        self.daily_users_categories = np.random.choice(list(self.customer_classes.keys()),
                                                       size=self.daily_clicks,
                                                       p=self.customer_distributions).tolist()

    def get_daily_user_features(self, index):
        # list of possible combination of the feature space
        cur_user_category = self.daily_users_categories[index]
        cur_user_features = self.customer_classes[cur_user_category]['features']
        idx = np.random.choice(len(cur_user_features))
        return cur_user_features[idx]
