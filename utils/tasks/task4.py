from typing import Dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from data_generators.data_generator import DataGenerator
from environments.contextual_environment import ContextualEnvironment
from learners.pricing.contextual_learner import ContextualLearner
from learners.pricing.thompson_sampling import ThompsonSampling
from learners.pricing.ucb import UCB
from utils.context_generator_v2 import ContextGenerator
from utils.tasks.task import Task


class Task4(Task):
    """
    Task representing the step 4: the bid is fixed and it is learnt in an online fashion the best pricing strategy
    when the algorithm discriminate among the customers’ classes.
    """

    def __init__(self, data_generator: DataGenerator, name="Step#4", description="", verbose=1):
        super().__init__(name, description, data_generator, verbose)
        # input data
        self.prices = data_generator.get_prices()
        self.bids = data_generator.get_bids()
        self.margins = data_generator.get_margins()
        self.classes = data_generator.get_classes()
        self.conversion_rates = data_generator.get_conversion_rates(mode='all')
        self.future_purchases = data_generator.get_future_purchases(mode='all')
        self.costs_per_click = data_generator.get_costs_per_click(mode='aggregate')
        self.n_clicks = data_generator.get_daily_clicks(mode='aggregate')
        self.selected_bid = 3
        self.fixed_cost = self.costs_per_click[self.selected_bid]
        self.fixed_n_clicks = np.rint(self.n_clicks[self.selected_bid]).astype(int)
        self.features = data_generator.get_features()
        # control variables
        self.bandit_args={
            'arm_values': self.margins
        }
        self.fractions = []
        for cl in data_generator.get_classes().values():
            self.fractions.append(cl['fraction'])

        self.cg_start_from = None
        self.cg_frequency = None
        self.cg_confidence = None

    def _serial_run(self, process_id: int, n_experiments: int, collector: Dict) -> None:
        rewards_per_experiment = {}
        context_split_per_experiment = {}

        for learner in self.learners_to_test:
            rewards_per_experiment[learner.LEARNER_NAME] = []
            context_split_per_experiment[learner.LEARNER_NAME] = []

        for e in range(n_experiments):
            # Initialization of the learners to test and their related environment:
            # the list is composed of tuples (Learner, Environment)
            self._print(f'core_{process_id}: running experiment {e + 1}/{n_experiments}...')
            test_instances = []
            for learner in self.learners_to_test:
                context_learner = ContextualLearner(self.features, learner, **self.bandit_args)
                test_instances.append(
                    (context_learner,
                     ContextualEnvironment(features=self.features,
                                           customer_classes=self.classes,
                                           conversion_rates=self.conversion_rates,
                                           future_purchases=self.future_purchases,
                                           daily_clicks=self.fixed_n_clicks,
                                           cost_per_click=self.fixed_cost),
                     ContextGenerator(features=self.features,
                                      contextual_learner=context_learner,
                                      update_frequency=self.cg_frequency,
                                      start_from=self.cg_start_from,
                                      confidence=self.cg_confidence,
                                      verbose=0))
                )

            for t in range(self.T):
                for context_learner, env, context_generator in test_instances:
                    context_learner.next_day()
                    past_arms = None
                    past_features = None
                    month_purchases = env.get_next_purchases_at_day(t, keep=False)
                    if month_purchases is not None:
                        past_arms = env.get_selected_arms_at_day(t - 30, keep=False)
                        past_features = env.get_collected_user_features_at_day(t - 30, keep=False)
                        context_learner.update_next_purchases(past_arms, month_purchases, past_features)

                    pulled_arms = context_learner.pull_arms()
                    daily_rewards = env.day_round(pulled_arms)
                    daily_users_features = env.get_collected_user_features_at_day(t)
                    daily_pulled_arms = env.get_selected_arms_at_day(t)

                    context_learner.update(daily_rewards, daily_pulled_arms, daily_users_features)
                    context_generator.collect_daily_data(daily_pulled_arms, daily_rewards, daily_users_features,
                                                         next_purchases=month_purchases, past_pulled_arms=past_arms,
                                                         past_features=past_features)

            for learner, _, _ in test_instances:
                learner.next_day()
                rewards_per_experiment[learner.base_learner_class.LEARNER_NAME].append(learner.get_daily_rewards())
                context_split_per_experiment[learner.base_learner_class.LEARNER_NAME].append(learner.get_splits_count())
        # end -> save rhe results.
        collector[process_id] = (rewards_per_experiment, context_split_per_experiment)

    def config(self,
               time_horizon: int,
               n_experiments: int,
               learner_to_test=None,
               cg_start_from=150,
               cg_frequency=10,
               cg_confidence=0.001,
               verbose=1) -> None:
        if learner_to_test is None:
            learner_to_test = [UCB, ThompsonSampling]
        self.T = time_horizon
        self.n_experiments = n_experiments
        self.learners_to_test = learner_to_test
        self.verbose = verbose
        self.cg_start_from = cg_start_from
        self.cg_frequency = cg_frequency
        self.cg_confidence = cg_confidence
        self.metadata['TIME_HORIZON'] = self.T
        self.metadata['NUMBER_OF_EXPERIMENTS'] = self.n_experiments
        self.metadata['CG_START_FROM'] = self.cg_start_from
        self.metadata['CG_CONFIDENCE'] = self.cg_confidence
        self.metadata['CG_FREQUENCY'] = self.cg_frequency

        self._print(f"{'*' * 20} ACTUAL CONFIGURATION {'*' * 20}")
        self._print(f'N_ROUNDS: {self.T}')
        self._print(f'N_EXPERIMENTS: {self.n_experiments}')
        self._print(f'ALGORITHMS: {[l.LEARNER_NAME for l in self.learners_to_test]}')
        self._print(f'\nSelected bid: {self.bids[self.selected_bid]}({self.selected_bid})')
        self._print(f'Fixed CPC: {self.fixed_cost}')
        self._print(f'Fixed num_clicks: {self.n_clicks[self.selected_bid]} -> {self.fixed_n_clicks}')
        self._print(f'Context generator: frequency={self.cg_frequency}, '
                    f'start_from={self.cg_start_from}, confidence={self.cg_confidence}\n')

    def load(self, filename):
        super(Task4, self).load(filename)
        self.T = self.metadata['TIME_HORIZON']
        self.n_experiments = self.metadata['NUMBER_OF_EXPERIMENTS']
        self.cg_start_from = self.metadata['CG_START_FROM']
        self.cg_frequency = self.metadata['CG_FREQUENCY']
        self.cg_confidence = self.metadata['CG_CONFIDENCE']

    def _finalize_run(self, collected_values: list) -> None:
        # set the result attribute
        rewards_dict = {}
        splits_dict = {}
        for learner in self.learners_to_test:
            rewards_dict[learner.LEARNER_NAME] = []
            splits_dict[learner.LEARNER_NAME] = []
        for rew_dict, spl_dict in collected_values:
            for learner in self.learners_to_test:
                for value in rew_dict[learner.LEARNER_NAME]:
                    rewards_dict[learner.LEARNER_NAME].append(value)
                for value in spl_dict[learner.LEARNER_NAME]:
                    splits_dict[learner.LEARNER_NAME].append(value)
        self.result['rewards'] = {}
        self.result['splits'] = {}
        for learner in self.learners_to_test:
            self.result['rewards'][learner.LEARNER_NAME] = np.mean(rewards_dict[learner.LEARNER_NAME], axis=0).tolist()
            self.result['splits'][learner.LEARNER_NAME] = splits_dict[learner.LEARNER_NAME]

    def plot(self, plot_number=0, figsize=(10, 8), theme='white') -> None:
        assert self.ready
        if plot_number < 0 or plot_number > 2:
            raise TypeError("`plot_number` kwarg error: only 2 plot are available.")

        sns.set_theme(style=theme)

        aggr_conv_rate = self.data_generator.get_conversion_rates(mode='aggregate')
        aggr_next_purchases = self.data_generator.get_future_purchases(mode='aggregate')
        aggr_opt = np.max(self.margins * aggr_conv_rate * (1 + aggr_next_purchases) - self.fixed_cost)

        opt_arms = []
        global_opt = 0
        for i, conv_rate in enumerate(self.conversion_rates):
            opt_arm = np.argmax(self.margins * conv_rate * (1 + self.future_purchases[i]) - self.fixed_cost)
            opt_value = np.max(self.margins * conv_rate * (1 + self.future_purchases[i]) - self.fixed_cost)
            opt_arms.append((opt_arm, opt_value))
            global_opt += self.fractions[i] * opt_value

        if plot_number == 0:
            plt.figure(0, figsize=figsize)
            plt.ylabel("Regret")
            plt.xlabel("Day")
            for val in self.result['rewards'].values():
                plt.plot(np.cumsum(global_opt * self.fixed_n_clicks - val))
            plt.legend(self.result['rewards'].keys())
            plt.title("Cumulative regret")
            plt.show()

        elif plot_number == 1:
            plt.figure(1, figsize=figsize)
            plt.xlabel("Day")
            plt.ylabel("Daily reward")
            plt.plot([global_opt * self.fixed_n_clicks] * self.T, '--g', label='clairvoyant')
            plt.plot([aggr_opt * self.fixed_n_clicks] * self.T, '--c', label='clairvoyant_aggregated')
            for key in self.result['rewards']:
                plt.plot(self.result['rewards'][key], label=key)
            plt.legend(loc='best')
            plt.title("Reward by day")
            plt.show()

        elif plot_number == 2:
            plt.figure(2, figsize=figsize)
            plt.ylabel("num_splits")
            plt.xlabel("experiment")
            print(self.result['splits'])
            for val in self.result['splits'].values():
                plt.plot(val, '--o')
            plt.legend(self.result['splits'].keys())
            plt.title("Number of splits")
            plt.show()
