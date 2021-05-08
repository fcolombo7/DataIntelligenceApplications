import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    """
    Abstract class used to represent online learning algorithms.
    """

    LEARNER_NAME = 'ABSTRACT LEARNER'

    def __init__(self, arm_values, period, next_purchases_update):
        """
        Class constructor
        :param arm_values: the action values of the arms that can be pulled
        :type arm_values: list
        :param period: the length of the window of time in which a customer can buy another unit at the same price
        :type period: int
        :param next_purchases_update: how the parameter of the distribution is estimated [Possible values: binomial]
        :type next_purchases_update: str
        """

        self.t = 0
        self.n_arms = len(arm_values)
        self.arm_values = arm_values
        self.outcome_per_arm = [[] for _ in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.daily_collected_rewards = np.array([])

        self.period = period
        self.next_purchases_estimation = np.zeros(self.n_arms)
        self.next_purchases_observations = [[] for _ in range(self.n_arms)]
        self.next_purchases_update = next_purchases_update

    def update_observations(self, pulled_arm, outcome, cost):
        """

        :param pulled_arm:
        :param outcome:
        :param cost:
        :return: void
        """
        self.outcome_per_arm[pulled_arm].append(outcome)
        actual_reward = outcome * self.arm_values[pulled_arm] * (1 + self.next_purchases_estimation[pulled_arm]) - cost
        self.collected_rewards = np.append(self.collected_rewards, actual_reward)

    def __binomial_update(self, pulled_arm, next_purchases):
        """
        Update the next_purchases distribution assuming a binomial distribution with the known parameter n = 30
        :param pulled_arm:
        :param next_purchases:
        :return:
        """
        n_observations = len(self.next_purchases_observations[pulled_arm])
        if n_observations == 0:
            self.next_purchases_estimation[pulled_arm] = next_purchases / self.period
        else:
            self.next_purchases_estimation[pulled_arm] = \
                (self.next_purchases_estimation[pulled_arm] * n_observations * self.period + next_purchases) / \
                (self.period * (n_observations + 1))

    @abstractmethod
    def pull_arm(self) -> int:
        """

        :return: void
        """
        pass

    @abstractmethod
    def update(self, pulled_arm, outcome, cost):
        """

        :param pulled_arm:
        :param outcome:
        :param cost:
        :return:
        """
        pass

    def daily_update(self, pulled_arm, daily_rew):
        """

        :param pulled_arm:
        :param daily_rew:
        :return:
        """
        r = daily_rew[:, 0] * self.arm_values[pulled_arm] * (1 + self.next_purchases_estimation[pulled_arm]*self.period) - daily_rew[:, 1]
        self.daily_collected_rewards = np.append(self.daily_collected_rewards, np.sum(r))
        for outcome, cost in daily_rew:
            self.update(pulled_arm, outcome, cost)

    def update_future_purchases(self, pulled_arm, daily_future_obs):
        if self.next_purchases_update == 'binomial':
            for ob in daily_future_obs:
                self.__binomial_update(pulled_arm, ob)
                self.next_purchases_observations[pulled_arm].append(ob)
        else:
            raise NotImplementedError()


