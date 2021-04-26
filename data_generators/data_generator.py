from abc import ABC, abstractmethod


class DataGenerator(ABC):
    """ Data generator interface. A DataGenerator instance must provide the following method to work properly
    alongside the Environment instance. """

    # TODO: check the actual methods and define new ones if necessary.

    @abstractmethod
    def get_prices(self):
        """ Get the available prices of a unit of a consumable item. """
        pass

    @abstractmethod
    def get_margin(self):
        """ Get the margin associated to each price. """
        pass

    @abstractmethod
    def get_bids(self):
        """ Get the available bids of a unit of a consumable item. """
        pass

    @abstractmethod
    def get_classes(self):
        """ Get the classes of customers. """
        pass

    @abstractmethod
    def get_conversion_rates(self):
        """ (For each class) Get the conversion rates distribution. """
        pass

    @abstractmethod
    def get_future_purchases(self):
        """ (For each class) Get the distribution probability over the number of times the user will come back to the
        ecommerce website to buy another consumable item by 30 days after the first purchase (at the same price). """
        pass

    @abstractmethod
    def get_daily_clicks(self):
        """(For each class) Get the probability distribution of the number of daily clicks of new users as a function
        of the bid. """
        pass

    @abstractmethod
    def get_costs_per_click(self):
        """(For each class) Get the probability distribution of the cost per click as a function of the bid."""
        pass
