from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_generators.standard_generator import StandardDataGenerator
from environments.contextual_environment import ContextualEnvironment
from environments.pricing_environment import PricingEnvironment
from learners.pricing.thompson_sampling import ThompsonSampling
from learners.pricing.ucb import UCB
from utils.tasks.task import Task
from data_generators.data_generator import DataGenerator


class Task3(Task):
    """
    Task representing the step 3: the bid is fixed and it is learnt in an online fashion the best pricing strategy
    when the algorithm does not discriminate among the customers’ classes.
    """

    def __init__(self, data_generator: DataGenerator, name="Step#3", description="", verbose=1):
        super().__init__(name, description, data_generator, verbose)
        # input data
        self.prices = data_generator.get_prices()
        self.bids = data_generator.get_bids()
        self.margins = data_generator.get_margins()
        self.selected_bid = 3
        self.conversion_rates = data_generator.get_conversion_rates(mode='all')
        self.future_purchases = data_generator.get_future_purchases(mode='all')
        self.n_clicks = data_generator.get_daily_clicks(mode='all')
        self.costs_per_click = data_generator.get_costs_per_click(mode='aggregate', bid=self.selected_bid)
        self.fixed_n_clicks = np.rint(data_generator.get_daily_clicks(mode='aggregate')[self.selected_bid]).astype(int)
        self.fixed_cost = self.costs_per_click[self.selected_bid]
        # control variables
        self.fractions = data_generator.get_class_distributions(bid=self.selected_bid)

        temp = (self.margins * np.average(self.conversion_rates * (1 + self.future_purchases),
                                          axis=0,
                                          weights=self.fractions) - self.fixed_cost) * self.fixed_n_clicks
        self.aggr_opt_arm = np.argmax(temp)
        self.aggr_opt = np.max(temp)

        self.env_params = {
            'bid_idx': self.selected_bid,
            # 'mode': 'aggregate',
            'mode': 'all',
            'src': data_generator.get_source(),
            'generator': 'standard'
        }

    def _serial_run(self, process_id: int, n_experiments: int, collector: Dict) -> None:
        rewards_per_experiment = {}
        for learner in self.learners_to_test:
            rewards_per_experiment[learner.LEARNER_NAME] = []

        for e in range(n_experiments):
            # Initialization of the learners to test and their related environment:
            # the list is composed of tuples (Learner, Environment)
            self._print(f'core_{process_id}: running experiment {e+1}/{n_experiments}...')
            test_instances = []
            for learner in self.learners_to_test:
                test_instances.append((learner(arm_values=self.margins),
                                       ContextualEnvironment(**self.env_params)))
                                       # PricingEnvironment(**self.env_params)))
            for t in range(self.T):
                for learner, env in test_instances:
                    learner.next_day()

                    past_arms = env.get_selected_arms_at_day(t - 30, keep=False, filter_purchases=True)
                    month_purchases = env.get_next_purchases_at_day(t, keep=False, filter_purchases=True)
                    if month_purchases is not None:
                        for arm, n_purchases in zip(past_arms, month_purchases):
                            learner.update_single_future_purchase(arm, n_purchases)

                    pulled_arm = learner.pull_arm()
                    daily_reward = env.day_round(pulled_arm)
                    for outcome, cost in daily_reward:
                        learner.update(pulled_arm, outcome, cost)

            for learner, _ in test_instances:
                learner.next_day()
                rewards_per_experiment[learner.LEARNER_NAME].append(learner.daily_collected_rewards)
        # end -> save rhe results.
        collector[process_id] = rewards_per_experiment

    def _finalize_run(self, collected_values: list):
        # set the result attribute
        aggregate_dict = {}
        for learner in self.learners_to_test:
            aggregate_dict[learner.LEARNER_NAME] = []
        for single_dict in collected_values:
            for key in single_dict:
                for value in single_dict[key]:
                    aggregate_dict[key].append(value)

        for learner in self.learners_to_test:
            self.result[learner.LEARNER_NAME] = np.mean(aggregate_dict[learner.LEARNER_NAME], axis=0).tolist()

    def config(self,
               time_horizon: int,
               n_experiments: int,
               learner_to_test=None,
               verbose=1):
        if learner_to_test is None:
            learner_to_test = [UCB, ThompsonSampling]
        self.T = time_horizon
        self.n_experiments = n_experiments
        self.learners_to_test = learner_to_test
        self.verbose = verbose
        self.metadata['TIME_HORIZON'] = self.T
        self.metadata['NUMBER_OF_EXPERIMENTS'] = self.n_experiments

        self._print(f"{'*'*20} ACTUAL CONFIGURATION {'*'*20}")
        self._print(f'N_ROUNDS: {self.T}')
        self._print(f'N_EXPERIMENTS: {self.n_experiments}')
        self._print(f'ALGORITHMS: {[l.LEARNER_NAME for l in self.learners_to_test]}')
        self._print(f'\nSelected bid: {self.bids[self.selected_bid]}({self.selected_bid})')
        self._print(f'Fixed CPC: {self.fixed_cost}')
        self._print(f'Fixed num_clicks: {self.fixed_n_clicks}\n')

    def load(self, filename):
        super(Task3, self).load(filename)
        self.T = self.metadata['TIME_HORIZON']
        self.n_experiments = self.metadata['NUMBER_OF_EXPERIMENTS']

    def plot(self, plot_number=0, figsize=(10, 8), theme="white"):
        assert self.ready
        if plot_number < 0 or plot_number > 1:
            raise TypeError("`plot_number` kwarg error: only 2 plot are available.")

        sns.set_theme(style=theme)

        if plot_number == 0:
            plt.figure(0, figsize=figsize)
            plt.ylabel("Regret")
            plt.xlabel("Day")
            for val in self.result.values():
                plt.plot(np.cumsum(self.aggr_opt - val))
            plt.plot(4000 * np.sqrt(np.linspace(0, 364, 364)), '--')
            labels = list(self.result.keys()) + ['sqrt(N)']
            plt.legend(labels)
            plt.title("Cumulative regret")
            plt.show()

        elif plot_number == 1:
            plt.figure(1, figsize=figsize)
            plt.xlabel("Day")
            plt.ylabel("Daily reward")
            plt.plot([self.aggr_opt] * self.T, '--g', label='clairvoyant')
            for key in self.result:
                plt.plot(self.result[key], label=key)
            plt.legend(loc='best')
            plt.title("Reward by day")
            plt.show()


# DEBUG - Not sure it works anymore due to changes
if __name__ == '__main__':
    task = Task3(StandardDataGenerator('../../src/basic004.json'))
    task.load(filename='../../simulations_results/result_Step#3_3.zip')
    task.plot(0)
    task.plot(1)

    """
    print('PARALLEL EXECUTION')
    start_time = time.perf_counter_ns()
    task = Task3(BasicDataGenerator('../../src/basic001.json'))
    # task.load('../../simulations_results/result_Step#3.zip')
    task.config(time_horizon=365, n_experiments=10, verbose=0)
    task.run(parallelize=True)
    task.save('../../simulations_results')
    end_time = time.perf_counter_ns()
    print(f"Execution time: {end_time - start_time} ns\n")
    print('SERIAL EXECUTION')
    start_time = time.perf_counter_ns()
    task = Task3(BasicDataGenerator('../../src/basic001.json'))
    # task.load('../../simulations_results/result_Step#3.zip')
    task.config(time_horizon=365, n_experiments=10, verbose=0)
    task.run(parallelize=False)
    task.save('../../simulations_results')
    end_time = time.perf_counter_ns()
    print(f"Execution time: {end_time - start_time} ns")
    """
