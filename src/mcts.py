import random
import numpy as np

# ============================================================================================================================ #
# Tree
# ============================================================================================================================ #

class Node(object):

    def __init__(self, state_manager, parent):
        self.parent = parent
        self.state_manager = state_manager
        self.moves = state_manager.get_moves()
        #
        self.children = []
        #
        self.n = {}
        self.p = {}
        self.v = {}
        self.q = {}
        self.u = {}
        self.w = {}

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state_manager.is_terminal_state()

    def tree_policy(self, child):
        return self.q[child] + self.u[child]

    def get_best_child(self):
        max_value = - np.inf
        candidates = []

        for child, value in zip(self.children, [self.tree_policy(child) for child in self.children]):
            if value == max_value:
                candidates.append(child)
            elif value > max_value:
                candidates = [child]
                max_value = value

        best_child = random.choice(candidates)
        return best_child

    def expand(self, network_wrapper):
        for move in self.moves:
            child = Node(state_manager = self.state_manager.make_move(self.moves.index(move)),
                parent = self)

            # this works, but we can do better...
            (p, v) = network_wrapper.forward(child.state_manager.state2vec())[0]

            self.v[child] = v
            self.p[child] = p
            self.w[child] = 0
            self.n[child] = 0
            self.q[child] = 0
            self.u[child] = 0
            self.children.append(child)
        return self

    def update(self, child, v):
        self.n[child] += 1.0
        self.w[child] += v
        self.q[child] = (1 / self.n[child]) * self.w[child]
        for child in self.children:
            # u will change for all children due to change in the summation of n[child]
            c_puct = 1.0
            # maybe some actual value for this constant?
            self.u[child] = c_puct * self.p[child] * ((sum(list(map(lambda x: self.n[x], self.children))) ** 0.5) / (1 + self.n[child]))

    def export_pi(self, move_number):
        temperature = (1.0 if move_number < 30 else 0.05) 
        return list(map(lambda x: self.n[x] ** (1.0 / temperature), self.children))

# ============================================================================================================================ #
# MCTS
# ============================================================================================================================ #

class MCTS(object):

    def __init__(self, network_wrapper):
        self.network_wrapper = network_wrapper

    def __call__(self, state_manager, n = 1500):
        root = Node(state_manager = state_manager, parent = None)
        print('root:', root)
        for i in range(0, 1500):
            node = self.traverse(root)
            self.back_propagate(node)
        return root.export_pi(state_manager.num_full_moves())

    def traverse(self, node):
        while not node.is_terminal():
            if node.is_leaf():
                return node.expand(self.network_wrapper)
            else:
                node = node.get_best_child()
        return node

    def back_propagate(self, node):
        # this is the value predicted for the edge leading to this node by the NN
        v = node.parent.v[node] if node.parent else 0
        while node.parent is not None:
            node.parent.update(child = node, v = v)
            node = node.parent
