import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from environments.complete_environment import CompleteEnvironment

from learners.pricing.contextual_learner import ContextualLearner
from utils.context_generator import ContextGenerator
from utils.tasks.task import Task


class CompleteTask(Task):
    """
    """

    def __init__(self, data_src: str, name: str, description="", fixed_adv=False, fixed_price=False,
                 selected_bid=4, selected_price=3, pricing_context=False, verbose=1):
        super().__init__(name, description, data_src, verbose)
        assert (fixed_adv is False and fixed_price is False) or (fixed_adv is not fixed_price)
        # task variable
        self.fixed_adv = fixed_adv
        self.selected_bid = selected_bid
        self.fixed_price = fixed_price
        self.selected_price = selected_price
        self.pricing_context = pricing_context

        # input data
        self.prices = self.data_generator.get_prices()
        self.bids = self.data_generator.get_bids()
        self.margins = self.data_generator.get_margins()
        self.classes = self.data_generator.get_classes()
        self.conversion_rates = self.data_generator.get_conversion_rates(mode='all')
        self.future_purchases = self.data_generator.get_future_purchases(mode='all')
        self.number_of_clicks = self.data_generator.get_daily_clicks(mode='all')
        self.costs_per_click = self.data_generator.get_costs_per_click(mode='aggregate', bid=self.selected_bid)
        self.features = self.data_generator.get_features()

        # control variables
        self.fractions = self.data_generator.get_class_distributions(self.selected_bid)
        self.bandit_args = {
            'arm_values': self.margins
        }
        self.cg_start_from = None
        self.cg_frequency = None
        self.cg_confidence = None

    def run(self) -> None:
        rewards_per_experiment = {}
        context_split_per_experiment = {}

        for learner in self.learners_to_test:
            rewards_per_experiment[learner.LEARNER_NAME] = []
            context_split_per_experiment[learner.LEARNER_NAME] = []

        for e in range(self.n_experiments):
            # Initialization of the learners to test and their related environment:
            # the list is composed of tuples (Learner, Environment)
            self._print(f'running experiment {e + 1}/{self.n_experiments}...')
            test_instances = []
            for learner_class in self.learners_to_test:
                env = CompleteEnvironment(self.data_src)
                learner = None
                context_generator = None
                if self.fixed_price:
                    # step 5
                    learner = learner_class(self.data_generator.get_bids())
                if self.fixed_adv:
                    if self.pricing_context:
                        # step 4
                        learner = ContextualLearner(self.features, learner_class, **self.bandit_args)
                        context_generator = ContextGenerator(features=self.features,
                                                             contextual_learner=learner,
                                                             update_frequency=self.cg_frequency,
                                                             start_from=self.cg_start_from,
                                                             confidence=self.cg_confidence,
                                                             verbose=0)
                    else:
                        # step 3
                        learner = learner_class(arm_values=self.data_generator.get_margins())
                if not self.fixed_adv and not self.fixed_price:
                    if self.pricing_context:
                        # step 7
                        learner = learner_class(margin_values=self.data_generator.get_margins(),
                                                bid_values=self.data_generator.get_bids(),
                                                features=self.features)
                        context_generator = ContextGenerator(features=self.features,
                                                             contextual_learner=learner,
                                                             update_frequency=self.cg_frequency,
                                                             start_from=self.cg_start_from,
                                                             confidence=self.cg_confidence,
                                                             verbose=0)
                    else:
                        # step 6
                        learner = learner_class(self.data_generator.get_margins(), self.data_generator.get_bids())
                test_instances.append((learner, env, context_generator))

            for t in range(self.T):
                for learner, env, context_generator in test_instances:
                    if self.fixed_price:
                        sel_bid = learner.pull_arm()
                        out = env.day_round(self.selected_price, sel_bid, fixed_price=True)
                        learner.update(sel_bid, out)
                    else:
                        learner.next_day()
                        past_arms = env.get_selected_arms_at_day(t - 30, keep=False, filter_purchases=True)
                        past_features = env.get_collected_user_features_at_day(t - 30, keep=False, filter_purchases=True)
                        month_purchases = env.get_next_purchases_at_day(t, keep=False, filter_purchases=True)
                        if month_purchases is not None:
                            if not self.pricing_context:
                                for arm, n_purchases in zip(past_arms, month_purchases):
                                    learner.update_single_future_purchase(arm, n_purchases)
                            else:
                                learner.update_next_purchases(past_arms, month_purchases, past_features)

                        if self.fixed_adv:
                            pulled_bid = self.selected_bid
                            pulled_arm = learner.pull_arm()
                        else:
                            pulled_bid, pulled_arm = learner.pull_arm()
                        # print(f"day: {t} - arm: {pulled_arm}, bid: {pulled_bid}")
                        daily_reward, sampled_n_clicks, sampled_cpc = env.day_round(pulled_arm,
                                                                                    selected_bid=pulled_bid,
                                                                                    fixed_adv=self.fixed_adv)
                        if not self.fixed_adv:
                            learner.get_adv_params(sampled_n_clicks, sampled_cpc)
                        if not self.pricing_context:
                            for outcome, cost in daily_reward:
                                learner.update(pulled_arm, outcome, cost)
                        else:
                            daily_users_features = env.get_collected_user_features_at_day(t)
                            daily_pulled_arms = env.get_selected_arms_at_day(t)
                            learner.update(daily_reward, daily_pulled_arms, daily_users_features)
                            context_generator.collect_daily_data(daily_pulled_arms,
                                                                 daily_reward,
                                                                 daily_users_features,
                                                                 next_purchases=month_purchases,
                                                                 past_pulled_arms=past_arms,
                                                                 past_features=past_features)

            for learner, _, _ in test_instances:
                if not self.fixed_price:
                    learner.next_day()
                if self.fixed_adv and self.pricing_context:
                    key = learner.base_learner_class.LEARNER_NAME
                else:
                    key = learner.LEARNER_NAME
                rewards_per_experiment[key].append(learner.get_daily_rewards())
                if self.pricing_context:
                    context_split_per_experiment[key].append(learner.get_splits_count())
        # end -> save rhe results.
        self.result['rewards'] = {}
        for key in rewards_per_experiment.keys():
            self.result['rewards'][key] = np.mean(rewards_per_experiment[key], axis=0).tolist()
        self.result['splits'] = context_split_per_experiment
        self._finalize_execution()

    def config(self,
               time_horizon: int,
               n_experiments: int,
               learner_to_test: list,
               cg_start_from=31,
               cg_frequency=10,
               cg_confidence=0.002,
               verbose=1) -> None:
        self.T = time_horizon
        self.n_experiments = n_experiments
        self.learners_to_test = learner_to_test
        self.verbose = verbose
        self.metadata['TIME_HORIZON'] = self.T
        self.metadata['NUMBER_OF_EXPERIMENTS'] = self.n_experiments
        if self.pricing_context:
            self.cg_start_from = cg_start_from
            self.cg_frequency = cg_frequency
            self.cg_confidence = cg_confidence
            self.metadata['CG_START_FROM'] = self.cg_start_from
            self.metadata['CG_CONFIDENCE'] = self.cg_confidence
            self.metadata['CG_FREQUENCY'] = self.cg_frequency

        self._print(f"{'*' * 20} ACTUAL CONFIGURATION {'*' * 20}")
        self._print(f'N_ROUNDS: {self.T}')
        self._print(f'N_EXPERIMENTS: {self.n_experiments}')
        self._print(f'ALGORITHMS: {[l.LEARNER_NAME for l in self.learners_to_test]}')
        if self.pricing_context:
            self._print(
                f'Context generator: frequency={self.cg_frequency}, start_from={self.cg_start_from}, confidence={self.cg_confidence}\n')

    def save(self, folder='simulations_results', overwrite=False) -> str:
        self.metadata['FIXED_ADV'] = self.fixed_adv
        self.metadata['FIXED_PRICE'] = self.fixed_price
        self.metadata['SELECTED_BID'] = self.selected_bid
        self.metadata['SELECTED_PRICE'] = self.selected_price
        self.metadata['PRICING_CONTEXT'] = self.pricing_context
        return super(CompleteTask, self).save(folder, overwrite)

    def load(self, filename):
        super(CompleteTask, self).load(filename)
        self.T = self.metadata['TIME_HORIZON']
        self.n_experiments = self.metadata['NUMBER_OF_EXPERIMENTS']
        self.fixed_adv = self.metadata['FIXED_ADV']
        self.fixed_price = self.metadata['FIXED_PRICE']
        self.selected_bid = self.metadata['SELECTED_BID']
        self.selected_price = self.metadata['SELECTED_PRICE']
        self.pricing_context = self.metadata['PRICING_CONTEXT']
        if self.pricing_context:
            self.cg_start_from = self.metadata['CG_START_FROM']
            self.cg_frequency = self.metadata['CG_FREQUENCY']
            self.cg_confidence = self.metadata['CG_CONFIDENCE']
        self._compute_opt_values()

    def plot(self, plot_number=0, figsize=(10, 8), theme="whitegrid") -> None:
        assert self.ready
        if plot_number < 0 or plot_number > 2:
            raise TypeError("`plot_number` kwarg error: only 3 plot are available.")

        sns.set_theme(style=theme)

        if plot_number == 0:
            plt.figure(0, figsize=figsize)
            plt.ylabel("Regret")
            plt.xlabel("Day")
            if self.pricing_context:
                opt = self.disaggr_opt
            else:
                opt = self.aggr_opt
            for val in self.result['rewards'].values():
                plt.plot(np.cumsum(opt - val))
            #plt.plot(1000*np.sqrt(np.linspace(0, 364, 365))+15000, '--')
            labels = list(self.result['rewards'].keys()) + ['O(sqrt(T))']
            plt.legend(labels)
            plt.title("Cumulative regret")
            plt.show()

        elif plot_number == 1:
            plt.figure(1, figsize=figsize)
            plt.xlabel("Day")
            plt.ylabel("Daily reward")
            plt.plot([self.disaggr_opt] * self.T, '--g', label='clairvoyant')
            plt.plot([self.aggr_opt] * self.T, '--c', label='aggr_clairvoyant')
            for key in self.result['rewards']:
                plt.plot(self.result['rewards'][key], label=key)
            plt.legend(loc='best')
            plt.title("Reward by day")
            plt.show()

        elif plot_number == 2:
            if not self.pricing_context:
                print("No splits to show. No context generator used in this task.")
                return
            plt.figure(2, figsize=figsize)
            plt.ylabel("num_splits")
            plt.xlabel("experiment")
            for val in self.result['splits'].values():
                plt.plot(val, '--o')
            plt.legend(self.result['splits'].keys())
            plt.title("Number of splits")
            plt.show()

    def _compute_opt_values(self):
        if self.fixed_adv:
            fixed_cost = self.costs_per_click[self.selected_bid]
            fixed_n_clicks = np.rint(self.data_generator.get_daily_clicks(mode='aggregate')[self.selected_bid]).astype(
                int)
            temp = (self.margins * np.average(self.conversion_rates * (1 + self.future_purchases),
                                              axis=0,
                                              weights=self.fractions) - fixed_cost) * fixed_n_clicks
            self.aggr_opt_arm = np.argmax(temp)
            self.aggr_opt = np.max(temp)
            self.disaggr_opt = 0
            self.opt_arms = []
            disaggr_costs = self.data_generator.get_costs_per_click(mode='all')
            for i, _ in enumerate(self.conversion_rates):
                t = (self.margins * self.conversion_rates[i] * (1 + self.future_purchases[i]) - disaggr_costs[
                    i, self.selected_bid]) * \
                    self.number_of_clicks[i, self.selected_bid]
                opt_arm = np.argmax(t)
                opt_value = np.max(t)
                self.opt_arms.append(opt_arm)
                self.disaggr_opt += opt_value
            return
        if self.fixed_price:
            expected_rewards = np.array([])
            for bid in range(0, len(self.data_generator.get_bids())):
                conv_rate = self.data_generator.get_conversion_rates(mode='aggregate', bid=bid)[self.selected_price]
                tau = self.data_generator.get_future_purchases(mode='aggregate', bid=bid)[self.selected_price]
                cpc = self.data_generator.get_costs_per_click(mode='aggregate', bid=bid)[bid]
                exp = self.data_generator.get_daily_clicks(mode='aggregate')[bid] * (
                        conv_rate * self.data_generator.get_margins()[self.selected_price] * (1 + tau) - cpc)
                expected_rewards = np.append(expected_rewards, exp)
            self.aggr_opt = np.max(expected_rewards)
            return
        if not self.fixed_price and not self.fixed_adv:
            aggr_arr = []
            disaggr_arr = []
            for j, bid in enumerate(self.data_generator.get_bids()):
                maxs = []
                f_cost = self.data_generator.get_costs_per_click(mode='aggregate', bid=j)[j]
                clicks = self.data_generator.get_daily_clicks(mode='aggregate')[j]
                conv = self.data_generator.get_conversion_rates()
                fp = self.data_generator.get_future_purchases()
                for i in range(len(self.data_generator.get_classes())):
                    maxs.append(np.max((conv[i] * (1 + fp[i]) * self.data_generator.get_margins() -
                                        self.data_generator.get_costs_per_click()[i, j]) *
                                       self.data_generator.get_daily_clicks()[i, j]))

                temp = (self.data_generator.get_margins() *
                        np.average(conv * (1 + fp), axis=0,
                                   weights=self.data_generator.get_class_distributions(bid=j)) - f_cost) * clicks
                # aggregated value for the current bid
                aggr_arr.append(np.max(temp))
                # disaggregated for the current bid
                disaggr_arr.append(sum(maxs))
            self.aggr_opt = max(aggr_arr)
            opt_bid = np.argmax(aggr_arr)
            self.disaggr_opt = max(disaggr_arr)
            opt_dis_bid = np.argmax(disaggr_arr)
            return
        raise NotImplementedError
