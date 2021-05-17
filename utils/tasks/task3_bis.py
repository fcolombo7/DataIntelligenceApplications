from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environments.pricing_environment_bis import PricingEnvironment
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
        self.conversion_rates = data_generator.get_conversion_rates(mode='aggregate')
        self.costs_per_click = data_generator.get_costs_per_click(mode='aggregate')
        self.n_clicks = data_generator.get_daily_clicks(mode='aggregate')
        self.future_purchases = data_generator.get_future_purchases(mode='aggregate')
        self.selected_bid = 3
        self.fixed_cost = self.costs_per_click[self.selected_bid]
        self.fixed_n_clicks = np.rint(self.n_clicks[self.selected_bid]).astype(int)
        # control variables
        self.opt_arm = np.argmax(self.margins * self.conversion_rates * (1 + self.future_purchases) -
                                 self.costs_per_click[self.selected_bid])

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
                                       PricingEnvironment(mode='aggregate',
                                                          src='src/basic001.json')))
            for t in range(self.T):
                for learner, env in test_instances:
                    learner.next_day()
                    month_purchases = env.get_next_purchases_at_day(t, keep=False)
                    if month_purchases is not None:
                        pulled_arms = env.get_selected_arms_at_day(t - 30, keep=False)
                        for arm, n_purchases in zip(pulled_arms, month_purchases):
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
        self._print(f'Fixed num_clicks: {self.n_clicks[self.selected_bid]} -> {self.fixed_n_clicks}\n')

    def load(self, filename):
        super(Task3, self).load(filename)
        self.T = self.metadata['TIME_HORIZON']
        self.n_experiments = self.metadata['NUMBER_OF_EXPERIMENTS']

    def plot(self, plot_number=0, figsize=(10, 8), theme="white"):
        assert self.ready
        if plot_number < 0 or plot_number > 1:
            raise TypeError("`plot_number` kwarg error: only 2 plot are available.")

        sns.set_theme(style=theme)

        opt = (self.margins[self.opt_arm] * self.conversion_rates[self.opt_arm] *
               (1 + self.future_purchases[self.opt_arm]) - self.costs_per_click[self.selected_bid]) * \
              np.rint(self.n_clicks[self.selected_bid]).astype(int)

        if plot_number == 0:
            plt.figure(0, figsize=figsize)
            plt.ylabel("Regret")
            plt.xlabel("Day")
            for val in self.result.values():
                plt.plot(np.cumsum(opt - val))
            plt.legend(self.result.keys())
            plt.title("Cumulative regret")
            plt.show()

        elif plot_number == 1:
            plt.figure(1, figsize=figsize)
            plt.xlabel("Day")
            plt.ylabel("Daily reward")
            plt.plot([opt] * self.T, '--g', label='clairvoyant')
            for key in self.result:
                plt.plot(self.result[key], label=key)
            plt.legend(loc='best')
            plt.title("Reward by day")
            plt.show()


# DEBUG - Not sure it works anymore due to changes
"""
if __name__ == '__main__':
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
