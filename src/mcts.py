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
        # the action parent took to arrive here
        self.move_in = None
        # initally all moves are untried
        self.untried_actions = list(map(lambda x: x, self.moves))
        # tried moves
        self.children = []
        # 
        self.v = 0
        self.n = {}
        self.q = {}
        self.p = {}

    def has_untried_actions(self):
        return not len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state_manager.is_terminal_state()

    def get_best_child(self, tree_policy):
        best_child = utils.rand_max(node.children.values(), policy = self.tree_policy)
        return best_child

    def expand(self, node):
        move = self.untried_actions[random.choice(range(0, len(self.untried_actions)))]
        child = Node(state_manager = self.state_manager.make_move(self.moves.index(move)),
            parent = self, move_in = move)
        self.n[child] = 0
        self.q[child] = 0
        self.p[child] = 0
        self.children.append(child)
        return child

    def update(self, child, v):
        self.v += v
        self.n[child] += 1.0
        self.q[child] = (1 / self.n[child]) * self.v

class MCTS(object):

    def __init__(self, tree_policy):
        self.tree_policy = tree_policy

    def __call__(self, state_manager, n = 1500):
        root = Node(state_manager, None)
        for i in range(n):
            node = self.get_next_node(root)
            self.back_propagate(node, self.evaluate_nn(node))
        # now that the simulation is done, return some node
        return root.get_best_child(self.tree_policy).move_in

    def get_next_node(self, node):
        while not node.is_terminal():
            if node.has_untried_actions:
                return node
            else:
                node = node.get_best_child(self.tree_policy)
        return node

    def back_propagate(self, node, v):
        while node is not None:
            node.parent.update(node, v)
            node = node.parent

    def evaluate_nn(self, node):
        # get this from the NN
        return 1.0
