import random
import numpy

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
        # initally all moves are untried
        self.untried_actions = list(map(lambda x: x, self.moves))
        # tried moves
        self.children = []
        # 
        self.v = 0
        self.n = {}
        self.q = {}
        self.u = {}
        self.p = {}
        self.visit_count = 0

    def has_untried_actions(self):
        return not len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state_manager.is_terminal_state()

    def get_best_child(self, tree_policy):
        max_value = - numpy.inf
        candidates = []

        for key, value in zip(self.children, [tree_policy(self, child) for child in self.children]):
            if value == max_value:
                candidates.append(key)
            elif value > max_value:
                candidates = [key]
                max_value = value

        best_child = random.choice(candidates)
        return best_child

    def expand(self):
        move = self.untried_actions[random.choice(range(0, len(self.untried_actions)))]
        child = Node(state_manager = self.state_manager.make_move(self.moves.index(move)),
            parent = self, move_in = move)
        self.n[child] = 0.0
        self.children.append(child)
        return child

    def update(self, child, v, move_number):
        self.visit_count += 1
        self.v += v
        self.n[child] += 1.0
        # maybe add 1 to both nominator and denominator to avoid zero?
        for child in self.children:
            self.p[child] = self.n[child] / (self.visit_count)
            self.q[child] = (1 / self.n[child]) * self.v
            c_puct = (1.0 if move_number < 30 else 0.05)
            self.u[child] = c_puct * self.p[child] * ((self.visit_count ** 0.5) / (1 + self.n[child]))

# ============================================================================================================================ #
# MCTS
# ============================================================================================================================ #

class MCTS(object):

    def __init__(self, tree_policy):
        self.tree_policy = tree_policy

    def __call__(self, state_manager, move_number, n = 1500):
        root = Node(state_manager = state_manager, parent = None, move_in = None)
        for i in range(n):
            node = self.get_next_node(root)
            self.back_propagate(node, self.evaluate_nn(node), move_number)
        # now that the simulation is done, return some node
        return root.get_best_child(self.tree_policy).move_in

    def get_next_node(self, node):
        # either expand 
        while not node.is_terminal():
            if node.has_untried_actions:
                return node.expand()
            else:
                node = node.get_best_child(self.tree_policy)
        return node

    def back_propagate(self, node, v, move_number):
        while node.parent is not None:
            node.parent.update(child = node, v = v, move_number = move_number)
            node = node.parent

    def evaluate_nn(self, node):
        # get this from the NN
        return 1.0
