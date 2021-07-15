from data_generators.data_generator import DataGenerator
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt


class StandardDataGenerator(DataGenerator):
    """
    `DataGenerator` implementations that reads the values and parameters from a json input source.
    :param filename: path to the json source
    :type filename: str
    """

    def __init__(self, filename):

        self._filename = filename
        self._data = None
        with open(self._filename) as f:
            self._data = json.load(f)
        # features
        self._features = self._data['features']
        # classes
        self._classes = {}
        # class distributions
        # self._class_distribution = []
        for key in self._data['classes']:
            self._classes[key] = {}
            # fraction = self._data['classes'][key]['fraction']
            # self._classes[key]['fraction'] = fraction
            # self._class_distribution.append(fraction)
            self._classes[key]['features'] = []  # the goal is to have a list of tuple, representing the subspace
            if not isinstance(self._data['classes'][key]['features'][0], list):
                features = self._data['classes'][key]['features']
                self._classes[key]['features'].append((features[0], features[1]))
            else:
                # TODO: Here i need a list of lists, but in doing the cast from the json the outer list is discarded.
                for features in self._data['classes'][key]['features']:
                    self._classes[key]['features'].append((features[0], features[1]))

        # prices
        self._prices = self._data['prices']
        # bids
        self._bids = self._data['bids']
        # margin
        self._margin = self._data['margins']
        # conversion rates
        self._conversion_rates = self._data['conversion_rates']
        # daily clicks
        self._daily_clicks = self._data['daily_clicks']
        # cost_per_click
        self._cost_per_click = self._data['cost_per_click']
        # future_purchases
        self._future_purchases = self._data['future_purchases']

    def get_source(self) -> str:
        return self._filename

    def get_all(self):
        """Return the content of the input source."""
        return self._data

    def get_features(self):
        return self._features

    def get_prices(self):
        """Return the available prices."""
        return self._prices

    def get_margins(self):
        """ Get the margin associated to each price. """
        return self._margin

    def get_bids(self):
        """ Get the available bids of a unit of a consumable item. """
        return self._bids

    def get_classes(self):
        """ Get the customer classes description as a list of dictionaries. """
        return self._classes

    def get_conversion_rates(self, mode='all', bid=None):
        """
        Get the conversion rates distribution. The output depends on the `mode` kwarg.
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")
        if mode == 'aggregate' and bid is None:
            raise TypeError("`bid` kwarg must be set if the mode is `aggregate`")

        if mode == 'all':
            return np.around(self._conversion_rates, decimals=3)
        class_distribution = self.get_class_distributions(bid)
        # return np.average(self._conversion_rates, axis=0, weights=class_distribution)
        return np.around(np.average(self._conversion_rates, axis=0, weights=class_distribution), decimals=3)

    def get_future_purchases(self, mode='all', bid=None):
        """
        Get the distribution probability over the number of times the user will come back to the
        ecommerce website to buy another consumable item by 30 days after the first purchase (at the same price).\n
        The output depends on the `mode` kwarg.\n
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        \nModel: np.maximum('lower_bound', 'coefficient'*(- prices + min(prices)) + 'upper_bound')
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")
        if mode == 'aggregate' and bid is None:
            raise TypeError("`bid` kwarg must be set if the mode is `aggregate`")

        future_purchases = []
        for cl in self._future_purchases:
            purchases = list(np.maximum(cl['lower_bound'],
                                        cl['coefficient']*(-np.array(self._prices) + self._prices[0]) + cl['upper_bound']))
            future_purchases.append(purchases)
        # TODO: CHECK HERE! devo mettere decimals = 0 poichè int? Forse no perchè questa è solo la media.
        if mode == 'all':
            return np.around(future_purchases, decimals=3)
        class_distribution = self.get_class_distributions(bid)
        # return np.average(future_purchases, axis=0, weights=class_distribution)
        return np.around(np.average(future_purchases, axis=0, weights=class_distribution), decimals=3)

    def get_daily_clicks(self, mode='all'):
        """
        Get the distribution probability over the number of daily clicks with respect to the bid of the advertisement
        campaign given the price.
        \nThe output depends on the `mode` kwarg.
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        \nModel: `upper_bound` * (1.0 - exp(-1 * `speed_factor` * `bids`))
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")

        daily_clicks = []
        for cl in self._daily_clicks:
            clicks_per_bid = list(cl['upper_bound'] * (1.0 - np.exp(-1 * cl['speed_factor'] * np.array(self._bids))))
            daily_clicks.append(clicks_per_bid)
        # TODO: CHECK HERE! CAMBIATO PER LA 3^ VOLTA.
        if mode == 'all':
            return np.around(daily_clicks, decimals=3)
        # return np.sum(daily_clicks, axis=0)
        return np.around(np.sum(daily_clicks, axis=0), decimals=3)

    def get_costs_per_click(self, mode='all', bid=None):
        """
        Get the distribution probability over the cost per click as a function of the bid
        \nThe output depends on the `mode` kwarg.
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        \nModel: `coefficient` * log(1 + `bids`/`coefficient`)
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")
        costs = []
        if mode == 'aggregate' and bid is None:
            raise TypeError("`bid` kwarg must be set if the mode is `aggregate`")

        for cl in self._cost_per_click:
            costs_per_bid = list(cl['coefficient'] * np.log(1 + np.array(self._bids)/cl['coefficient']))
            costs.append(costs_per_bid)
        if mode == 'all':
            return np.around(costs, decimals=3)
        class_distribution = self.get_class_distributions(bid)
        # return np.average(costs, axis=0, weights=class_distribution)
        return np.around(np.average(costs, axis=0, weights=class_distribution), decimals=3)

    def get_class_distributions(self, bid) -> []:
        daily_clicks = self.get_daily_clicks(mode='all')
        clicks_at_bid = [x[bid] for x in daily_clicks]
        return np.array(clicks_at_bid)/np.sum(clicks_at_bid)


class Plotter:
    def __init__(self, filename):
        self.filename = filename
        self.dg = StandardDataGenerator(filename)
        self.categories = list(self.dg.get_classes().keys())

    def plot_conversion_rates(self, aggregate=False, sel_bid=5, figsize=(6, 4), theme='whitegrid', sub=False):
        sns.set_theme(style=theme)
        prices = self.dg.get_prices()
        if sub:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.set_xticks(prices)
            ax2.set_xticks(prices)
        else:
            ax1 = ax2 = None
            plt.figure(figsize=figsize)
            plt.xticks(prices)
        if not aggregate or sub:
            conv_rates = self.dg.get_conversion_rates()
            for i in range(0, len(conv_rates)):
                if sub:
                    ax1.plot(prices, conv_rates[i], '-o', label=self.categories[i])
                    ax1.set_title("Conversion rates")
                    ax1.legend(loc='best')
                else:
                    plt.plot(prices, conv_rates[i], '-o', label=self.categories[i])
                    plt.title("Conversion rates")
                    plt.legend(loc='best')
        if aggregate or sub:
            conv_rate = self.dg.get_conversion_rates(mode='aggregate', bid=sel_bid)
            if sub:
                ax2.plot(prices, conv_rate, '-o', label='aggr. conv. rate')
                ax2.set_title("Aggregated conversion rate")
                ax2.legend(loc='best')
            else:
                plt.plot(prices, conv_rate, '-o', label='aggr. conv. rate')
                plt.title("Aggregated conversion rate")
                plt.legend(loc='best')
        plt.show()

    def plot_daily_clicks(self, aggregate=False, figsize=(6, 4), theme='whitegrid', sub=False):
        sns.set_theme(style=theme)
        bids = self.dg.get_bids()
        if sub:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.set_xticks(bids)
            ax2.set_xticks(bids)
            ax1.tick_params('x', labelrotation=70)
            ax2.tick_params('x', labelrotation=70)
        else:
            ax1 = ax2 = None
            plt.figure(figsize=figsize)
            plt.xticks(bids, rotation=70)
        if not aggregate or sub:
            daily_clicks = self.dg.get_daily_clicks()
            for i in range(0, len(daily_clicks)):
                if sub:
                    ax1.plot(bids, daily_clicks[i], '-o', label=self.categories[i])
                    ax1.set_title("Daily clicks")
                    ax1.legend(loc='best')
                else:
                    plt.plot(bids, daily_clicks[i], '-o', label=self.categories[i])
                    plt.title("Daily clicks")
                    plt.legend(loc='best')
        if aggregate or sub:
            daily_clicks = self.dg.get_daily_clicks(mode='aggregate')
            if sub:
                ax2.plot(bids, daily_clicks, '-o', label='aggr. daily clicks')
                ax2.set_title("Aggregated daily clicks")
                ax2.legend(loc='best')
            else:
                plt.plot(bids, daily_clicks, '-o', label='aggr. daily clicks')
                plt.title("Aggregated daily clicks")
                plt.legend(loc='best')
        plt.show()

    def plot_costs_per_clicks(self, aggregate=False, sel_bid=5, figsize=(6, 4), theme='whitegrid', sub=False):
        sns.set_theme(style=theme)
        bids = self.dg.get_bids()
        if sub:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.set_xticks(bids)
            ax2.set_xticks(bids)
            ax1.tick_params('x', labelrotation=70)
            ax2.tick_params('x', labelrotation=70)
            ax1.plot(bids, bids, color='black', label="y=x")
            ax2.plot(bids, bids, color='black', label="y=x")
        else:
            ax1 = ax2 = None
            plt.figure(figsize=figsize)
            plt.xticks(bids, rotation=70)
            plt.plot(bids, bids, color='black', label="y=x")
        if not aggregate or sub:
            costs = self.dg.get_costs_per_click()
            for i in range(0, len(costs)):
                if sub:
                    ax1.plot(bids, costs[i], '-o', label=self.categories[i])
                    ax1.set_title("Costs per click")
                    ax1.legend(loc='best')
                else:
                    plt.plot(bids, costs[i], '-o', label=self.categories[i])
                    plt.title("Costs per click")
                    plt.legend(loc='best')
        if aggregate or sub:
            costs = self.dg.get_costs_per_click(mode='aggregate', bid=sel_bid)
            if sub:
                ax2.plot(bids, costs, '-o', label='aggr. cpc')
                ax2.set_title("Aggregated cost per click")
                ax2.legend(loc='best')
            else:
                plt.plot(bids, costs, '-o', label='aggr. cpc')
                plt.title("Aggregated cost per click")
                plt.legend(loc='best')
        plt.show()

    def plot_next_purchases(self, aggregate=False, sel_bid=5, figsize=(6, 4), theme='whitegrid', sub=False):
        sns.set_theme(style=theme)
        prices = self.dg.get_prices()
        if sub:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.set_xticks(prices)
            ax2.set_xticks(prices)
        else:
            ax1 = ax2 = None
            plt.figure(figsize=figsize)
            plt.xticks(prices)
        if not aggregate or sub:
            next_purch = self.dg.get_future_purchases()
            for i in range(0, len(next_purch)):
                if sub:
                    ax1.plot(prices, next_purch[i], '-o', label=self.categories[i])
                    ax1.set_title("Future purchases")
                    ax1.legend(loc='best')
                else:
                    plt.plot(prices, next_purch[i], '-o', label=self.categories[i])
                    plt.title("Future purchases")
                    plt.legend(loc='best')
        if aggregate or sub:
            next_purch = self.dg.get_future_purchases(mode='aggregate', bid=sel_bid)
            if sub:
                ax2.plot(prices, next_purch, '-o', label='aggr. tau')
                ax2.set_title("Aggregated future purchases")
                ax2.legend(loc='best')
            else:
                plt.plot(prices, next_purch, '-o', label='aggr. tau')
                plt.title("Aggregated future purchases")
                plt.legend(loc='best')
        plt.show()
