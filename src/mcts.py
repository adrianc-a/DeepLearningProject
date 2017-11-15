from __future__ import print_function

import random
from . import utils
import numpy as np

class MCTS(object):

    def __init__(self, policy):
        self.policy = policy
        self.gamma = 1.0

    def __call__(self, root, n = 1500):
        for i in range(n):
            node = _get_next_node(root, self.policy)
            node.reward = self.policy(node)
            self.back_propagate(node)
        return rand_max(root.children.values(), key=lambda x: x.q).action

    def back_propagate(not):
        r = node.reward
        while node is not None:
            node.n += 1
            node.q = ((node.n - 1) / node.n) * node.q + 1 / node.n * r
            node = node.parent

    def expand(node):
        action = random.choice(node.untried_actions)
        return node.children[action].sample_state()

    def best_child(node, tree_policy):
        best_child = utils.rand_max(node.children.values(), policy = tree_policy)
        return best_child.sample_state()

    def get_next_node(node, tree_policy):
        while not node.state.is_terminal():
            if node.untried_actions:
                return expand(node)
            else:
                state_node = best_child(node, tree_policy)
        return state_node

    def rand_max(iterable, policy = None):
        if key is None:
            key = lambda x: x

        max_value = -np.inf
        candidates = []

        for item, value in zip(iterable, [key(i) for i in iterable]):
            if value == max_value:
                candidates.append(item)
            elif value > max_value:
                candidates = [item]
                max_value = value

        return random.choice(candidates)
