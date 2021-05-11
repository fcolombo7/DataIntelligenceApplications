import numpy as np


class ContextualEnvironment:

    def __init__(self, customer_classes: {}, conversion_rates: [], future_purchases: [], daily_clicks: int,
                 cost_per_click: float):

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
        self.t = 0

    # todo: at first call get_user_features, then round [this is due to the cur.user update]
    def round(self, pulled_arm):
        # current user category
        self.daily_users_features.append(self.get_current_user_features())
        cur_user_category = self.daily_users_categories[self.cur_user]
        # cur_user_category is the index 'C1'
        cur_user_category_id = list(self.customer_classes.keys()).index(cur_user_category)
        self.cur_user += 1
        # determine the outcome of the experiment: buy / not but
        outcome = np.random.binomial(1, self.conversion_rates[cur_user_category_id][pulled_arm])
        if outcome != 0:
            p = self.future_purchases[cur_user_category_id][pulled_arm] / 30
            single_future_purchases = np.random.binomial(30, p)
            self.collected_future_purchases[self.t].append(single_future_purchases)
        return [outcome, self.cost_per_click]

    def day_round(self, pulled_arm):
        self.collected_future_purchases[self.t] = []
        self.selected_arms[self.t] = pulled_arm
        self.sample_daily_users()
        daily_rew = None
        for _ in range(self.daily_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = np.vstack((daily_rew, self.round(pulled_arm)))
        self.collected_users_features[self.t] = self.daily_users_features
        # increment the day counter
        self.t += 1
        # set the default daily categories/features
        self.daily_users_categories = None
        self.daily_users_features = None
        self.cur_user = None

        return daily_rew

    def get_future_purchases(self, day):  # todo: how to distinguish for customer class ?
        if day < 30:
            return None, []
        return self.selected_arms.pop(day - 30), self.collected_future_purchases.pop(day - 30), self.collected_users_features.pop(day - 30)

    def get_collected_user_features(self, day):
        return self.collected_users_features[day]

    def sample_daily_users(self):
        self.cur_user = 0
        self.daily_users_features = []
        self.daily_users_categories = np.random.choice(list(self.customer_classes.keys()),
                                                       size=self.daily_clicks,
                                                       p=self.customer_distributions).tolist()

    def get_current_user_features(self):
        # list of possible combination of the feature space
        cur_user_category = self.daily_users_categories[self.cur_user]
        cur_user_features = self.customer_classes[cur_user_category]['features']
        idx = np.random.choice(len(cur_user_features))
        return cur_user_features[idx]
