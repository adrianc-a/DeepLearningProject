import random
import numpy as np

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

class Node(object):

    def __init__(self, state_manager, parent):
        self.parent = parent
        self.state_manager = state_manager
        self.moves = state_manager.get_moves()
        self.reward = 0
        # initally all moves are untried
        self.untried_actions = list(map(lambda x: x, self.moves))
        self.children = []
        self.q = 0
        self.n = 0
        # self.children = list(map(lambda move_index: self.state_manager.make_move(move_index),
        #     range(0, len(self.moves))))

    def has_untried_actions(self):
        return not len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state_manager.is_terminal_state()

    def get_best_child(self, tree_policy):
        best_child = utils.rand_max(node.children.values(), policy = self.tree_policy)
        return best_child

    def expand(self, node):
        move_index = random.choice(range(0, len(self.moves)))
        child = Node(self.state_manager.make_move(move_index))
        self.children.append(child)
        return child

class MCTS(object):

    def __init__(self, tree_policy, default_policy):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.gamma = 1.0

    def __call__(self, state_manager, n = 1500):
        root = Node(state_manager, None)
        for i in range(n):
            node = self.get_next_node(root)
            node.reward = self.policy(node)
            self.back_propagate(node)
        # return rand_max(root.children.values(), key = lambda x: x.q).action

    def get_next_node(node):
        while not node.is_terminal():
            if node.has_untried_actions:
                return self.expand(node)
            else:
                node = node.get_best_child(self.tree_policy)
        return node

    def back_propagate(node):
        r = node.reward
        while node is not None:
            node.n += 1
            node.q = ((node.n - 1) / node.n) * node.q + 1 / node.n * r
            node = node.parent
