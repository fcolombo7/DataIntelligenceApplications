import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    """
    Abstract class used to represent online learning algorithms.
    :param n_arms: the number of arms that can be pulled
    :type n_arms: int
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        Update the collected reward of the pulled arm, and the total reward achieved.
        :param pulled_arm: the arm that was pulled
        :param reward: the reward received by the environment
        :return: void
        """
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    @abstractmethod
    def pull_arm(self) -> int:
        """
        Compute the best arm to pull according to the current parameter of the learner.

        :return:  the best arm to pull
        """
        pass

    @abstractmethod
    def update(self, pulled_arm, reward):
        """
        Update the parameters of the learner according the the reward received by the environment, and the pulled arm.

        :param pulled_arm: the arm that was pulled
        :param reward: the reward received by the environment
        :return: void
        """
        pass
