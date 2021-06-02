import copy
import numpy as np

from learners.pricing.contextual_learner import ContextualLearner
from learners.pricing.learner import Learner


def _print(text, verbose):
    if verbose > 0:
        print(text)


class TreeNode:
    """
    class representing the single node of the Context Tree
    """

    def __init__(self, all_features, base_learner):
        """
        Class constructor. Each node has it own learner, and a description of the feature space that covers.
        :param all_features: all the dimensions of the space
        :param base_learner: learner
        """
        self.left_child: TreeNode = None
        self.right_child: TreeNode = None
        self.all_features: dict = all_features
        self.base_learner: Learner = base_learner
        # dict[feature_name] = True/False
        self.feature_subspace: dict = {}

    def __str__(self):
        s = f'feature_subspace={self.feature_subspace} - is_leaf={self.is_leaf()}'
        return s

    def is_leaf(self) -> bool:
        """
        Chek if the the node is a leaf or not
        """
        return self.left_child is None and self.right_child is None

    def can_growth(self) -> bool:
        """
        Check if the node already covers all the available features.
        """
        return len(self.feature_subspace.keys()) < len(self.all_features)

    def get_leaves(self):
        """ Recursive method that returns the leaves of the tree. """
        # base case of the recursion
        if self.is_leaf():
            return [self]
        # otherwise this is not a leaf node, so check the child
        left_leaves = self.left_child.get_leaves()
        right_leaves = self.right_child.get_leaves()
        # concatenation of the children' leaves
        return left_leaves + right_leaves

    def get_daily_rewards(self):
        """ Recursive method that returns the leaves of the tree. """
        reward = np.append(np.array([]), self.base_learner.daily_collected_rewards)
        if not self.is_leaf():
            children_rewards = self.left_child.get_daily_rewards() + self.right_child.get_daily_rewards()
            reward = np.append(reward, children_rewards)
        return reward

    def split(self, splitting_feature, left_learner, right_learner):
        """
        Method used to actually create a new context.
        :param splitting_feature: the feature according to which the split is performed
        :param left_learner: the learner of the left child
        :param right_learner: the learner of the right child
        """
        # check if you can split using the arg: it is possible only if it is not already present in the feature_subspace
        assert splitting_feature in self.all_features and splitting_feature not in list(self.feature_subspace.keys()), \
            f"Cannot split the current node using `{splitting_feature}` as feature."
        # use deepcopy to get a child object that does not interfere with the parent one
        self.left_child = TreeNode(self.all_features, left_learner)
        self.left_child.feature_subspace = copy.deepcopy(self.feature_subspace)
        self.left_child.feature_subspace[splitting_feature] = False
        # right node --> feature = True
        self.right_child = TreeNode(self.all_features, right_learner)
        self.right_child.feature_subspace = copy.deepcopy(self.feature_subspace)
        self.right_child.feature_subspace[splitting_feature] = True
        # collect the daily reward of the parent node to effectively count in the overall reward
        self.base_learner.next_day()


class ContextGenerator:
    """
    Class representing the Context generation algorithm: this object receives the data generated by the environment and
    determines how and when it is worth to generate a new Context.
    """

    def __init__(self, features: [], contextual_learner: ContextualLearner, update_frequency: int, start_from: int,
                 confidence: float, verbose=1):
        """
        Class constructor
        :param features: all the features considered.
        :param contextual_learner: learner that is capable of handling contexts. It is used an observer pattern, so the
        context generator keeps a reference of the overall learner.
        :param update_frequency: frequency at which the context generation checks the data and run its generation algorithm.
        :param confidence: parameters used to determine the lower bound.
        """
        self.verbose = verbose
        self.features = features
        self.collected_arms = np.array([], dtype=np.int)
        self.collected_rewards = None
        self.collected_features = None

        self.collected_next_purchases = np.array([], dtype=np.int)
        self.collected_past_pulled_arms = np.array([], dtype=np.int)
        self.collected_past_features = []

        self.t = 0

        self.contextual_learner = contextual_learner
        self.update_frequency = update_frequency
        self.start_from = start_from
        self.confidence = confidence

        self.context_tree = TreeNode(features, self.contextual_learner.get_root_learner())
        self._update_contextual_learner()

        self.metadata = {
            'START_FROM': start_from,
            'FREQUENCY': update_frequency,
            'CONFIDENCE': confidence,
            'SPLIT_FEATURES': [],
            'SPLIT_DAYS': [],
            'LOG': '',
        }

    # TODO: HERE!!!!
    def collect_daily_data(self, pulled_arms, rewards, features,
                           next_purchases=None, past_pulled_arms=None, past_features=None):
        """
        Collect the data produced by the environment in one day
        :param pulled_arms: arms that are pulled at day t
        :param rewards: rewards collected at day t
        :param features: features of the users that play an arm at day t
        :param next_purchases: data about the number of times a user bought again the item in the past 30 days,
                               after the first purchase
        :param past_features: data about the users that pulled arms 30 days before.
        :param past_pulled_arms: data about the arms that was pulled 30 days before.
        """
        self.collected_arms = np.append(self.collected_arms, pulled_arms)
        # up to now it is a pair outcome - cost
        if self.collected_rewards is None:
            self.collected_rewards = rewards
        else:
            self.collected_rewards = np.vstack((self.collected_rewards, rewards))
        if self.collected_features is None:
            self.collected_features = features
        else:
            self.collected_features = np.vstack((self.collected_features, features))

        if next_purchases is not None:
            self.collected_next_purchases = np.append(self.collected_next_purchases, next_purchases)
        if past_pulled_arms is not None:
            self.collected_past_pulled_arms = np.append(self.collected_past_pulled_arms, past_pulled_arms)
        if past_features is not None:
            self.collected_past_features = self.collected_past_features + past_features
        self._context_generation()
        self.t += 1

    def _context_generation(self):
        """
        Algorithm that evaluates the possibility of generating a new context.
        """
        if self.t % self.update_frequency != 0 or self.start_from > self.t:
            return
        # get all the leaves that can be further expanded
        leaves = []
        for leaf in self.context_tree.get_leaves():
            if leaf.can_growth():
                leaves.append(leaf)
        if len(leaves) == 0:
            return
        # check if it is worth to split the leaves
        _print(f'\n{"-"*20} RUNNING CONTEXT GENERATOR@t={self.t} {"-"*20}', self.verbose)
        _print(f'N_LEAVES: {len(leaves)}', self.verbose)
        for leaf in leaves:
            self._evaluate_split(leaf)  # for each leaf evaluate if it is worth to split

    def _evaluate_split(self, leaf: TreeNode):
        """
        Check is the leaf can be split.
        :param leaf: leaf to split.
        """
        _print(f"- Evaluating the Node: {leaf}", self.verbose)

        best_feature = None
        features, values_after_split, right_learners, left_learners = self._iterate_over_features(leaf)
        # now get the max value after the split and the index
        max_value = max(values_after_split)
        idx = values_after_split.index(max_value)
        # print(f'{max_value=} [{idx=}]')
        # check if the value is larger than the value before split
        before_learner = leaf.base_learner
        value_before = self._compute_lower_bound(before_learner.get_opt_arm_expected_value()[0],
                                                 len(before_learner.collected_rewards))
        _print(f'\tValues after the split: {values_after_split}', self.verbose)
        _print(f'\tValue before the split: {value_before}\n', self.verbose)
        if value_before < max_value:
            best_feature = features[idx]
            # there is a feature for which it is worth to split
            _print(f'\t{best_feature=}', self.verbose)
            leaf.split(best_feature, left_learners[idx], right_learners[idx])
            _print(f'NEW CONTEXT GENERATED:\n splitting into -> {leaf.left_child.feature_subspace} and {leaf.right_child.feature_subspace}', self.verbose)
            self.metadata['SPLIT_DAYS'].append(self.t)
            s = str(leaf.feature_subspace) + ' -> ' + best_feature
            self.metadata['SPLIT_FEATURES'].append(s)
            self._log(leaf)
            self._update_contextual_learner()

    def _iterate_over_features(self, leaf):
        """
        Chek all the features that are not already expanded by the leaf and compute the related values after the split.
        The caller will check the maximum one and if it is greater than the value before split,
        the new context will be generated
        :param leaf: the leaf to be evaluated
        """
        values_after_split = []
        right_learners = []
        left_learners = []
        # get the features that are not expanded
        available_features = list(set(leaf.all_features) - set(leaf.feature_subspace.keys()))
        _print(f'\nFeatures to check: {available_features}', self.verbose)
        for feature in available_features:
            # compute the probability that the split happens
            _print(f'\nAnalysis of the feature `{feature}`...', self.verbose)
            feature_id = self.features.index(feature)
            check_condition = [None for _ in self.features]
            # print(f'{check_condition=} - {len(check_condition)=}')
            for f in leaf.feature_subspace:
                check_condition[self.features.index(f)] = leaf.feature_subspace[f]
            # here all the indices of the values compliant with the current feature space
            indices = []
            left_split_indices = []
            right_split_indices = []
            # collect the indices of the samples that are compliant with the selected feature
            for idx, collected_feature in enumerate(self.collected_features):
                cond = True
                i = 0
                while cond and i < len(check_condition):
                    if check_condition[i] is None:
                        i += 1
                        continue
                    if check_condition[i] != collected_feature[i]:
                        cond = False
                    i += 1
                if cond:
                    indices.append(idx)
                    if not collected_feature[feature_id]:
                        left_split_indices.append(idx)
                    else:
                        right_split_indices.append(idx)

            # GREEDY ALGORITHM.
            assert len(set(left_split_indices).union(set(right_split_indices)).difference(set(indices))) == 0
            left_split_probability = len(left_split_indices) / len(indices)
            right_split_probability = 1.0 - left_split_probability

            # get the left and right learners, trained with the data collected up to now
            left_subspace = copy.deepcopy(leaf.feature_subspace)
            right_subspace = copy.deepcopy(leaf.feature_subspace)
            left_subspace[feature] = False
            right_subspace[feature] = True
            left_learner = self._get_offline_trained_lerner(pulled_arms=self.collected_arms[left_split_indices],
                                                            rewards=self.collected_rewards[left_split_indices, 0],
                                                            costs=self.collected_rewards[left_split_indices, 1],
                                                            subspace=left_subspace)
            right_learner = self._get_offline_trained_lerner(pulled_arms=self.collected_arms[right_split_indices],
                                                             rewards=self.collected_rewards[right_split_indices, 0],
                                                             costs=self.collected_rewards[right_split_indices, 1],
                                                             subspace=right_subspace)
            left_value = left_learner.get_opt_arm_expected_value()[0]
            right_value = right_learner.get_opt_arm_expected_value()[0]
            value_after = self._compute_lower_bound(left_split_probability, len(left_split_indices)) * \
                          self._compute_lower_bound(left_value, len(left_learner.collected_rewards)) + \
                          self._compute_lower_bound(right_split_probability, len(right_split_indices)) * \
                          self._compute_lower_bound(right_value, len(right_learner.collected_rewards))

            values_after_split.append(value_after)
            right_learners.append(right_learner)
            left_learners.append(left_learner)
        return available_features, values_after_split, right_learners, left_learners

    def _update_contextual_learner(self):
        """
        Method used to update the tree of the contextual learner.
        """
        self.contextual_learner.update_context_tree(self.context_tree)

    def _compute_lower_bound(self, mean, n_samples):
        """
        Method used to compute the lower bound [Hoeffding-Bound]
        :param mean: mean of the distribution
        :param n_samples: cardinality
        """
        if n_samples == 0:
            return -np.inf
        ret_value = mean - np.sqrt(-np.log(self.confidence) / (2 * n_samples))
        return ret_value

    def _get_offline_trained_lerner(self, pulled_arms, rewards, costs, subspace):
        """
        Train a new learner to be set as base learner of a new context
        :param pulled_arms: history of pulled arms
        :param rewards: history of rewards received by the environment
        :param costs: history of received costs
        :param subspace: feature subspace of the learner
        :return:
        """
        learner = self.contextual_learner.get_root_learner()

        for a, r, c in zip(pulled_arms, rewards, costs):
            learner.update(a, r, c)
        # update the estimation of the ext purchase distribution
        counter = 0
        for i, purchases in enumerate(self.collected_next_purchases):
            update = True
            for f in subspace.keys():
                f_id = self.features.index(f)
                if self.collected_past_features[i][f_id] != subspace[f]:
                    update = False
                    break
            if update:
                counter += 1
                learner.update_single_future_purchase(self.collected_past_pulled_arms[i], purchases)
        return learner

    def _log(self, leaf):
        s = '\n'+15*'*'+f' SPLIT@day={self.t} '+15*'*'+'\n'
        s += f'{leaf.feature_subspace} -> {leaf.left_child.feature_subspace} AND {leaf.right_child.feature_subspace}\n'
        s += f'COLLECTED DATA:\n' \
             f'{self.collected_arms=}\n' \
             f'{self.collected_rewards=}\n' \
             f'{self.collected_features=}\n' \
             f'{self.collected_next_purchases=}\n' \
             f'{self.collected_past_pulled_arms=}\n' \
             f'{self.collected_past_features=}\n'
        s += f'{leaf.left_child.base_learner.daily_collected_rewards=}\n' \
             f'{leaf.left_child.base_learner.next_purchases_estimation=}\n'
        self.metadata['LOG'] += s
