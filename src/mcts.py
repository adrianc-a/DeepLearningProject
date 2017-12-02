import random
import numpy as np

class Node(object):

    def __init__(self, state_manager, parent, idx = -1):
        self.parent = parent
        self.state_manager = state_manager
        self.moves = state_manager.get_moves()

        num_children = len(self.moves)

        self.children = []

        self.n = np.zeros(num_children)
        self.p = np.zeros(num_children)
        self.v = 0
        self.q = np.zeros(num_children)
        self.u = np.zeros(num_children)
        self.w = np.zeros(num_children)
        self.idx = idx

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state_manager.is_terminal_state()

    def tree_policy(self, child):
        return self.q[child] + self.u[child]

    def get_best_child(self):
        ind = np.argmax(self.u + self.q)

        return self.children[ind], ind

    def expand(self, network_wrapper):
        #self.value = 0.0

        # get the children state managers and their vec representations
        state_vecs, state_mans = self.state_manager.moves2vec()

        #get the predicted p and v values for all the children
        p, v = network_wrapper.forward(state_vecs)
        self.v = np.mean(v)

        '''
        print('p')
        print(p)
        print('v')
        print(v)
        '''
        self.p = p
        for i, (_, next_state) in enumerate(zip(p, state_mans)):
            child = Node(state_manager = next_state,
                         parent = self, idx = i)
            self.w[i] = self.v
            self.n[i] = 1
            self.q[i] = self.w[i]
            self.children.append(child)
        return self

    def update(self, child_idx, v):
        self.n[child_idx] += 1.0
        self.w[child_idx] += v
        self.q[child_idx] = self.w[child_idx]/ self.n[child_idx]
        children_visits = np.sum(self.n)
        c_puct = 1.0
        for i,_ in enumerate(self.children):
            # u will change for all children due to change in the summation of n[child]
            # maybe some actual value for this constant?
            self.u[i] = c_puct * self.p[i] * np.sqrt(children_visits) / (1 + self.n[child_idx])

    def export_pi(self, move_number, temp_change_iter, temp_early, temp_late):
        temperature = temp_early if move_number < temp_change_iter else temp_late
        #return list(map(lambda x: self.n[x] ** (1.0 / temperature), self.children))
        temp_vals = np.power(self.n, 1 / temperature)

        return temp_vals / np.sum(temp_vals)

class MCTS(object):

    def __init__(self, network_wrapper, manager, temp_change_iter=30, temp_early=1, temp_late = 0.33):
        self.network_wrapper = network_wrapper
        self.manager = manager
        self.temp_change_iter = temp_change_iter
        self.temp_early = temp_early
        self.temp_late = temp_late
        self._begin_game()

    def _begin_game(self):
        self.root = Node(state_manager = self.manager.current_state(), parent = None)

    def __call__(self, state_manager, n = 1500):
        for i in range(n):
            (node, terminal) = self.traverse(self.root)
            if not terminal:
                self.back_propagate(node)

        # print('root: ', self.root)
        # print('children: ', self.root.children)
        return self.root.export_pi(state_manager.num_full_moves(),
                                  self.temp_change_iter,
                                  self.temp_early,
                                  self.temp_late)

    def set_root(self, move_idx):
        self.root = self.root.children[move_idx]

    def traverse(self, node):
        while not node.is_terminal():
            if node.is_leaf():
                return (node.expand(self.network_wrapper), False)
            else:
                node, ind = node.get_best_child()
                node.parent.update(ind, 0)

        return (node, True)

    def back_propagate(self, node):
        # this is the value predicted for the edge leading to this node by the NN
        #v = node.calc_value()
        while node.parent is not None:
            node.parent.update(child_idx = node.idx, v = node.v)
            node = node.parent
