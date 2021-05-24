import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    """
    Abstract class used to represent online learning algorithms.
    """

    LEARNER_NAME = 'ABSTRACT LEARNER'

    def __init__(self, arm_values, period=365, next_purchases_update='binomial'):
        """
        Class constructor
        :param arm_values: the action values of the arms that can be pulled
        :type arm_values: list
        :param period: the length of the window of time in which a customer can buy another unit at the same price
        :type period: int
        :param next_purchases_update: how the parameter of the distribution is estimated [Possible values: binomial]
        :type next_purchases_update: str
        """

        self.t = 0  # t is the counter of update
        self.day = -1  # day is the counter of days
        self.n_arms = len(arm_values)
        self.arm_values = arm_values
        self.outcome_per_arm = [[] for _ in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.daily_collected_rewards = np.array([])
        self.daily_rewards = []  # auxiliary variable that stores the rewards collected in one day

        self.period = period
        self.next_purchases_estimation = np.zeros(self.n_arms)
        self.next_purchases_observations = [[] for _ in range(self.n_arms)]
        self.next_purchases_update = next_purchases_update

    def __str__(self):
        num_samples = 0
        for arm in range(self.n_arms):
            num_samples += len(self.next_purchases_observations[arm])
        s = f'{self.day=};\n' \
            f'{self.next_purchases_estimation=};\n' \
            f'num_samples={num_samples};\n' \
            f'num_outcomes_per_arm={[len(self.outcome_per_arm[arm]) for arm in range(self.n_arms)]};\n'
        return s

    def update_observations(self, pulled_arm, outcome, cost):
        """
        If cost == -1 the learner is solving a bidding related problem
        :param pulled_arm:
        :param outcome:
        :param cost:
        :return: void
        """
        self.outcome_per_arm[pulled_arm].append(outcome)
        if cost == -1:
            actual_reward = outcome
        else:
            actual_reward = outcome * self.arm_values[pulled_arm] * (1 + self.next_purchases_estimation[pulled_arm]) - cost
            self.collected_rewards = np.append(self.collected_rewards, actual_reward)
        self.daily_rewards.append(actual_reward)

    def __binomial_update(self, pulled_arm, next_purchases):
        """
        Update the next_purchases distribution assuming a binomial distribution with the known parameter n = 30
        :param pulled_arm:
        :param next_purchases:
        :return:
        """
        n_observations = len(self.next_purchases_observations[pulled_arm])
        if n_observations == 0:
            self.next_purchases_estimation[pulled_arm] = next_purchases
        else:
            self.next_purchases_estimation[pulled_arm] = \
                (self.next_purchases_estimation[pulled_arm] * n_observations + next_purchases) / (n_observations + 1)

    def next_day(self) -> None:
        """ Increment the day counter """
        if self.day != -1:
            self.daily_collected_rewards = np.append(self.daily_collected_rewards, np.sum(self.daily_rewards))
        # reset the collector variable
        self.daily_rewards = []
        self.day += 1

    def update_single_future_purchase(self, pulled_arm, single_obs):  # TODO: new method used in the contextual version
        """
        Update the next purchases observation using a single observation of arm-value pair.
        :param pulled_arm: arm pulled 30 days in the past
        :param single_obs: number of purchases done in the past 30 days
        """
        if self.next_purchases_update == 'binomial':
            self.__binomial_update(pulled_arm, single_obs)
            self.next_purchases_observations[pulled_arm].append(single_obs)
        else:
            raise NotImplementedError()

    def get_next_purchases_data(self):
        """
            Method used to get the data about the estimation of the next purchases distributions.
            """
        return self.next_purchases_estimation, self.next_purchases_observations, self.next_purchases_update

    def set_next_purchases_data(self, estimation, observations, update_mode):
        """
            Method used to get data about previous computations of the estimates of the next purchases.
            This method is used when the context generation create a new learner.
            """
        self.next_purchases_estimation = estimation
        self.next_purchases_observations = observations
        self.next_purchases_update = update_mode

    @abstractmethod
    def pull_arm(self) -> int:
        pass

    @abstractmethod
    def update(self, pulled_arm, outcome, cost):
        """
            :param pulled_arm:
            :param outcome:
            :param cost:
            """
        pass



"""
def update_future_purchases(self, pulled_arm, daily_future_obs): # todo: check the step 3, probably can be changed and this method can be removed
    if self.next_purchases_update == 'binomial':
        for ob in daily_future_obs:
            self.__binomial_update(pulled_arm, ob)
            self.next_purchases_observations[pulled_arm].append(ob)
    else:
        raise NotImplementedError()
"""
"""   
    def daily_update(self, pulled_arm, daily_rew):
        # TODO: mean, not the parameter of the binomial -->
        #  r = daily_rew[:, 0] * self.arm_values[pulled_arm] * (1 + self.next_purchases_estimation[pulled_arm]*self.period) - daily_rew[:, 1]
        r = daily_rew[:, 0] * self.arm_values[pulled_arm] * (1 + self.next_purchases_estimation[pulled_arm]) - daily_rew[:, 1]
        self.daily_collected_rewards = np.append(self.daily_collected_rewards, np.sum(r))
        for outcome, cost in daily_rew:
            self.update(pulled_arm, outcome, cost)
    """