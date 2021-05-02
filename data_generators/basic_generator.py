from data_generators.data_generator import DataGenerator
import numpy as np
import json


class BasicDataGenerator(DataGenerator):
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
        # classes
        self._classes = self._data['classes']
        self._class_distribution = []
        # class distributions
        for cl in self._classes:
            self._class_distribution.append(cl['fraction'])
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

    def get_filename(self):
        """Return the path to the JSON input source."""
        return self._filename

    def get_all(self):
        """Return the content of the input source."""
        return self._data

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

    def get_conversion_rates(self, mode='all'):
        """
        Get the conversion rates distribution. The output depends on the `mode` kwarg.
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")
        if mode == 'all':
            return np.around(self._conversion_rates, decimals=3)
        return np.around(np.average(self._conversion_rates, axis=0, weights=self._class_distribution), decimals=3)

    def get_future_purchases(self, mode='all'):
        """
        Get the distribution probability over the number of times the user will come back to the
        ecommerce website to buy another consumable item by 30 days after the first purchase (at the same price).\n
        The output depends on the `mode` kwarg.\n
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        \nModel: np.maximum('lower_bound', 'coefficient'*(- prices + min(prices)) + 'upper_bound')
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")
        future_purchases = []
        for cl in self._future_purchases:
            purchases = list(np.maximum(cl['lower_bound'],
                                        cl['coefficient']*(-np.array(self._prices) + self._prices[0]) + cl['upper_bound']))
            future_purchases.append(purchases)
        # TODO: CHECK HERE! devo mettere decimals = 0 poichè int? Forse no perchè questa è solo la media.
        if mode == 'all':
            return np.around(future_purchases, decimals=3)
        return np.around(np.average(future_purchases, axis=0, weights=self._class_distribution), decimals=3)

    def get_daily_clicks(self, mode='all'):
        """
        Get the distribution probability over the number of times the user will come back to the
        ecommerce website to buy another consumable item by 30 days after the first purchase (at the same price).
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
        # TODO: CHECK HERE! devo mettere decimals = 0 poichè int? Forse no perchè questa è solo la media.
        if mode == 'all':
            return np.around(daily_clicks, decimals=3)
        return np.around(np.average(daily_clicks, axis=0, weights=self._class_distribution), decimals=3)

    def get_costs_per_click(self, mode='all'):
        """
        Get the distribution probability over the cost per click as a function of the bid
        \nThe output depends on the `mode` kwarg.
        [options: mode = `all` -> disjoint (default), mode = `aggregate` -> aggregation performed as a weighted average]
        \nModel: `coefficient` * log(1 + `bids`/`coefficient`)
        """
        if mode != 'all' and mode != 'aggregate':
            raise TypeError("`mode` kwarg error: the only valid choices are `all`(default) and `aggregate`")
        costs = []
        for cl in self._cost_per_click:
            costs_per_bid = list(cl['coefficient'] * np.log(1 + np.array(self._bids)/cl['coefficient']))
            costs.append(costs_per_bid)
        if mode == 'all':
            return np.around(costs, decimals=3)
        return np.around(np.average(costs, axis=0, weights=self._class_distribution), decimals=3)


# DEBUG
if __name__ == '__main__':
    dg = BasicDataGenerator("../src/basic001.json")
    print(dg.get_daily_clicks(mode='aggregate'))
