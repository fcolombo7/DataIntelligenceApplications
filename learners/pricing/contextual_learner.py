from learners.pricing.learner import Learner


class ContextualLearner:

    def __init__(self, features, base_learner_class: Learner.__class__, **base_learner_args):
        self.features = features
        self.base_learner_class = base_learner_class
        self.base_learner_args = base_learner_args
        # definition of the datastructures that handle the contexts
        self.context_tree = None

    def update_context_tree(self, new_context_tree):
        self.context_tree = new_context_tree

    def get_root_learner(self):
        return self.base_learner_class(**self.base_learner_args)

    def get_learner_by_context(self, current_features):
        # TODO: probably a faster way can be achieved by using the get_leaves method
        # navigation of the tree up to the leaf with the correct context
        cur_node = self.context_tree
        navigate = True
        while navigate:
            if cur_node.is_leaf():
                navigate = False
                break
            left_subspace = cur_node.left_child.feature_subspace
            go_left = True
            for feature in left_subspace:
                feature_idx = self.features.index(feature)
                if current_features[feature_idx] != left_subspace[feature]:
                    go_left = False
                    break
            if go_left:
                cur_node = cur_node.left_child
            else:
                # optional check: it should be the right child by construction
                right_subspace = cur_node.right_child.feature_subspace
                go_right = True
                for feature in left_subspace:
                    feature_idx = self.features.index(feature)
                    if current_features[feature_idx] != right_subspace[feature]:
                        go_right = False
                        break
                if go_right:
                    cur_node = cur_node.right_child
                else:
                    raise NotImplementedError("An error occurs: neither the left and the right child are compliant "
                                              "with the given features.")
        return cur_node.base_learner

    def update_next_purchases(self, pulled_arms_data: list, next_purchases_data: list, features_data: list):
        """
        :param pulled_arms_data: list containing all the pulled arms
        :param next_purchases_data: list containing the number of times a user returns in the next 30 days
        :param features_data: list of tuples representing the realization of the feature space
        """
        leaves = self.context_tree.get_leaves()
        # scan the data received by the environment and update the proper learner according to the context
        for i, obs in enumerate(next_purchases_data):
            # i -> index used to scan the data received by the environment
            for j, leaf in enumerate(leaves):
                # j -> index used to keep track of the correct leaf
                leaf_subspace = leaf.feature_subspace
                good_leaf = True
                for feature in leaf_subspace:
                    feature_idx = self.features.index(feature)
                    if features_data[i][feature_idx] != leaf_subspace[feature]:
                        good_leaf = False
                        break
                if good_leaf:
                    leaf.base_learner.update_single_future_purchase(pulled_arms_data[i], obs)

    def pull_arms(self):
        """ get a structure of arm to pull according to the context """
        context_arm_data = []
        for leaf in self.context_tree.get_leaves():
            context_arm_data.append((leaf.feature_subspace, leaf.base_learner.pull_arm()))
        return context_arm_data

    def update(self, daily_reward, pulled_arms, user_features):
        # scan and divide according to the features
        leaves = self.context_tree.get_leaves()
        dispatched_data = [[] for _ in leaves]

        # TODO: c'è da cambiare il learner !!!!  daily update non si può più usare ?

        for i, obs in enumerate(daily_reward):
            # i -> index used to scan the data received by the environment
            for j, leaf in enumerate(leaves):
                # j -> index used to keep track of the correct leaf
                leaf_subspace = leaf.feature_subspace
                good_leaf = True
                for feature in leaf_subspace:
                    feature_idx = self.features.index(feature)
                    if features_data[i][feature_idx] != leaf_subspace[feature]:
                        good_leaf = False
                        break
                if good_leaf:
                    leaf.base_learner.update_single_future_purchase(pulled_arms_data[i], obs)
