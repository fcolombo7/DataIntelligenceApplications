from learners.pricing.learner import Learner
from utils.context_generator import TreeNode


class ContextualLearner:

    def __init__(self, features, base_learner_class: Learner.__class__, **base_learner_args):
        self.features = features
        self.base_learner_class = base_learner_class
        self.base_learner_args = base_learner_args
        # definition of the datastructures that handle the contexts
        self.context_tree: TreeNode = None

    def update_context_tree(self, new_context_tree):
        self.context_tree = new_context_tree

    def get_root_learner(self):
        return self.base_learner_class(**self.base_learner_args)

    def get_learner_by_context(self, current_features):
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
