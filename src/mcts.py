import random
import numpy as np

# ============================================================================================================================ #
# Tree
# ============================================================================================================================ #

class Node(object):

    def __init__(self, state_manager, parent, move_in):
        self.parent = parent
        self.state_manager = state_manager
        self.moves = state_manager.get_moves()
        # the action parent took to arrive here
        self.move_in = move_in
        # tried moves
        self.children = []
        #
        self.value = 0
        #
        self.n = {}
        self.p = {}
        self.v = {} # ??
        self.q = {}
        self.u = {}
        self.w = {}

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state_manager.is_terminal_state()

    def get_best_child(self, tree_policy):
        max_value = - np.inf
        candidates = []

        # this produces a key_error
        for key, value in zip(self.children, [tree_policy(self, child) for child in self.children]):
            if value == max_value:
                candidates.append(key)
            elif value > max_value:
                candidates = [key]
                max_value = value

        best_child = random.choice(candidates)
        return best_child

    def expand(self, network_wrapper):
        self.value = 0.0

        # get the children state managers and their vec representations
        state_vecs, state_mans = self.moves2vec()

        #get the predicted p and v values for all the children
        p, v = network_wrapper.forward(state_vecs)

        for p, v, next_state, move_in in zip(p, v, state_mans, self.moves):
            child = Node(state_manager = next_state,
                         parent = self, move_in = move_in)

            self.v[child] = v
            self.p[child] = p
            self.w[child] = 0
            self.n[child] = 0
            self.q[child] = 0
            self.u[child] = 0
            self.children.append(child)
        return self

    def update(self, child, v):
        self.value += v
        self.n[child] += 1.0
        # maybe add 1 to both nominator and denominator to avoid zero?
        for child in self.children:
            self.q[child] = 0 if self.n[child] == 0 else (1 / self.n[child]) * self.value
            self.u[child] = self.p[child] * ((sum(list(map(lambda x: self.n[x], self.children))) ** 0.5) / (1 + self.n[child]))

    def export_pi(self, move_number):
        temperature = (1.0 if move_number < 30 else 0.05)
        # does this do anything?
        #s = sum(list(map(lambda x: self.n[x], self.children)))
        # what are we mapping over?
        return list(map(lambda x: self.n[x] ** (1.0 / temperature), self.children))

# ============================================================================================================================ #
# MCTS
# ============================================================================================================================ #

class MCTS(object):

    def __init__(self, tree_policy, network_wrapper):
        self.tree_policy = tree_policy
        self.network_wrapper = network_wrapper

    def __call__(self, state_manager, move_number, n = 5):
        root = Node(state_manager = state_manager, parent = None, move_in = None)
        for i in range(n):
            node = self.traverse(root)
            self.back_propagate(node)
        # now that the simulation is done, return some node
        return root.export_pi(move_number)

    def traverse(self, node):
        # either expand
        while not node.is_terminal():
            if node.is_leaf():
                return node.expand(self.network_wrapper)
            else:
                node = node.get_best_child(self.tree_policy)
        return node

    def back_propagate(self, node):
        v = node.parent.v[node] if node.parent else 0
        while node.parent is not None:
            node.parent.update(child = node, v = v)
            node = node.parent

    def evaluate_nn(self, node):
        # get this from the NN
        return 1.0
