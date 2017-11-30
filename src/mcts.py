import random
import numpy as np

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
        self.value = 0.0

        # get the children state managers and their vec representations
        state_vecs, state_mans = self.state_manager.moves2vec()

        #get the predicted p and v values for all the children
        p, v = network_wrapper.forward(state_vecs)
        avg_v = np.mean(v)

        for p, next_state in zip(p, state_mans):
            child = Node(state_manager = next_state,
                         parent = self)
            self.v[child] = 0
            self.p[child] = p
            self.w[child] = 0
            self.n[child] = 0
            self.q[child] = 0
            self.u[child] = 0
            self.children.append(child)
        return avg_v

    def update(self, child, v):
        self.n[child] += 1.0
        self.w[child] += v
        self.q[child] = (1 / self.n[child]) * self.w[child]
        s = sum(list(map(lambda x: self.n[x], self.children)))
        for child in self.children:
            # u will change for all children due to change in the summation of n[child]
            c_puct = 1.0
            # maybe some actual value for this constant?
            self.u[child] = c_puct * self.p[child] * (s ** 0.5) / (1 + self.n[child])

    def export_pi(self, move_number):
        temperature = (1.0 if move_number < 30 else 0.05)
        return list(map(lambda x: self.n[x] ** (1.0 / temperature), self.children))

    def calc_value(self):
        v = 0.0
        for child in self.children:
            v += self.v[child]
        return v / len(self.children)

# ============================================================================================================================ #
# MCTS
# ============================================================================================================================ #

class MCTS(object):

    def __init__(self, network_wrapper):
        self.network_wrapper = network_wrapper
        self.root = None

    def __call__(self, state_manager, n = 1500):
        if not self.root:
            # print('starting mcts simulation ...')
            self.root = Node(state_manager = state_manager, parent = None)
            # print('root is:', self.root)
        else:
            # state_manager.output()
            self.update_root(state_manager)
        for i in range(0, 1500):
            (node, terminal) = self.traverse(self.root)
            if not terminal:
                self.back_propagate(node)
        # print('root: ', self.root)
        # print('children: ', self.root.children)
        return self.root.export_pi(state_manager.num_full_moves())

    def update_root(self, state_manager):
        if not str(state_manager.board) == str(self.root.state_manager.board):
            for child in self.root.children:
                if str(child.state_manager.board) == str(state_manager.board):
                    return child
        else:
            return self.root
        print('PROBLEM')
        root = self.root
        while root.parent:
            root = root.parent
        return root
        # print('updating root')
        # go all the way to the tree root
        # root = self.root
        # while root.parent:
        #     root = root.parent
        # search for a state equal to this
        # self.root = self.search(root, state_manager)

    def search(self, root, state_manager):
        frontier = [root]
        while True:
            node = frontier.pop(0)
            if str(node.state_manager.board) == str(state_manager.board):
                return node
            for child in node.children:
                frontier.append(child)
            if not len(frontier):
                return None

    def make_move(self, move_index):
        self.root = self.root.children[move_index] 
        # print('root now at', self.root)

    def traverse(self, node):
        while not node.is_terminal():
            if node.is_leaf():
                return (node.expand(self.network_wrapper), False)
            else:
                node = node.get_best_child()
        return (node, True)

    def back_propagate(self, node, v):
        # this is the value predicted for the edge leading to this node by the NN
        v = node.calc_value()
        while node.parent is not None:
            node.parent.update(child = node, v = v)
            node = node.parent
