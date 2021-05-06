import numpy as np
import matplotlib.pyplot as plt

from data_generators.basic_generator import BasicDataGenerator
from environments.pricing_environment2 import PricingEnvironment
from learners.pricing.thompson_sampling import ThompsonSampling
from learners.pricing.ucb import UCB
from utils.tasks.task import Task
from data_generators.data_generator import DataGenerator


class Task3(Task):
    """

    """

    def __init__(self, data_generator: DataGenerator, name="Step#3", description=""):
        """

        :param data_generator:
        :param name:
        :param description:
        """
        super().__init__(name, description)
        self.data_generator = data_generator
        # input data
        self.prices = data_generator.get_prices()
        self.bids = data_generator.get_bids()
        self.margins = data_generator.get_margins()
        self.conversion_rates = data_generator.get_conversion_rates(mode='aggregate')
        self.costs_per_click = data_generator.get_costs_per_click(mode='aggregate')
        self.n_clicks = data_generator.get_daily_clicks(mode='aggregate')
        self.future_purchases = data_generator.get_future_purchases(mode='aggregate')
        self.selected_bid = 3
        # control variables
        self.opt_arm = np.argmax(self.margins * self.conversion_rates * (1 + self.future_purchases) -
                                 self.costs_per_click[self.selected_bid])
        self.T = None
        self.n_experiments = None
        self.learners_to_test = [UCB, ThompsonSampling]
        # result
        self.result = {}

    def run(self, force=False):
        if not force and self.ready:
            self._print('Warning: The task was already executed, so the result is available.')
            return
        if force and self.ready:
            self._print('Warning: Forcing the execution of the task, even if the result is available.')
        self._print(f'The execution of the task `{self.name}` is started.')
        # initialization of auxiliary variablesRunning
        fixed_cost = self.costs_per_click[self.selected_bid]
        fixed_n_clicks = np.rint(self.n_clicks[self.selected_bid]).astype(int)
        rewards_per_experiment = {}
        for learner in self.learners_to_test:
            rewards_per_experiment[learner.LEARNER_NAME] = []

        self._print(f'N_ROUNDS: {self.T}')
        self._print(f'N_EXPERIMENTS: {self.n_experiments}')
        self._print(f'ALGORITHMS: {list(rewards_per_experiment.keys())}')
        self._print(f'\nSelected bid: {self.bids[self.selected_bid]}({self.selected_bid})')
        self._print(f'Fixed CPC: {fixed_cost}')
        self._print(f'Fixed num_clicks: {self.n_clicks[self.selected_bid]} -> {fixed_n_clicks}')
        for e in range(self.n_experiments):
            # Initialization of the learners to test and their related environment:
            # the list is composed of tuples (Learner, Environment)
            self._print(f'running exp#{e}...')
            test_instances = []
            for learner in self.learners_to_test:
                test_instances.append((learner(arm_values=self.margins),
                                       PricingEnvironment(n_arms=len(self.prices),
                                                          conversion_rates=self.conversion_rates,
                                                          cost_per_click=fixed_cost,
                                                          n_clicks=fixed_n_clicks,
                                                          tau=self.future_purchases)))
            for t in range(self.T):
                for learner, env in test_instances:
                    pulled_arm = learner.pull_arm()
                    daily_reward = env.day_round(pulled_arm)
                    learner.daily_update(pulled_arm, daily_reward)

            for learner, _ in test_instances:
                rewards_per_experiment[learner.LEARNER_NAME].append(learner.daily_collected_rewards)

        # set the result attribute
        for learner in self.learners_to_test:
            self.result[learner.LEARNER_NAME] = np.mean(rewards_per_experiment[learner.LEARNER_NAME], axis=0).tolist()

        self._finalize_execution()

    def config(self,
               time_horizon,
               n_experiments,
               verbose=1):
        """

        :param time_horizon:
        :param n_experiments:
        :param verbose:
        :return:
        """
        self.T = time_horizon
        self.n_experiments = n_experiments
        self.verbose = verbose
        self.metadata['TIME_HORIZON'] = self.T
        self.metadata['NUMBER_OF_EXPERIMENTS'] = self.n_experiments

    def load(self, filename):
        super(Task3, self).load(filename)
        self.T = self.metadata['TIME_HORIZON']
        self.n_experiments = self.metadata['NUMBER_OF_EXPERIMENTS']

    def plot(self, plot_number=0, figsize=(10, 8)):
        assert self.ready
        if plot_number < 0 or plot_number > 1:
            raise TypeError("`plot_number` kwarg error: only 2 plot are available.")
        opt = (self.margins[self.opt_arm] * self.conversion_rates[self.opt_arm] *
               (1 + self.future_purchases[self.opt_arm]) - self.costs_per_click[self.selected_bid]) * \
              np.rint(self.n_clicks[self.selected_bid]).astype(int)

        if plot_number == 0:
            plt.figure(0, figsize=figsize)
            plt.ylabel("Regret")
            plt.xlabel("day")
            for val in self.result.values():
                plt.plot(np.cumsum(opt - val))
            plt.legend(self.result.keys())
            plt.title = "Cumulative regret"
            plt.show()

        elif plot_number == 1:
            plt.figure(1, figsize=figsize)
            plt.xlabel("day")
            plt.ylabel("Daily reward")
            plt.plot([opt] * self.T, '--', label='clairvoyant')
            for key in self.result:
                plt.plot(self.result[key], label=key)
            plt.legend(loc='best')
            plt.title = "Reward by day"
            plt.show()


# DEBUG
if __name__ == '__main__':
    task = Task3(BasicDataGenerator('../../src/basic001.json'))
    task.load('../../simulations_results/result_Step#3.zip')
    task.config(365, 50)
    task.run()
    task.save('../../simulations_results')
    task.plot(plot_number=0)
    task.plot(plot_number=1)
