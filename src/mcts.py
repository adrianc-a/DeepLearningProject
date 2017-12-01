import random
import numpy as np

class Node(object):

    def __init__(self, state_manager, parent, idx = -1):
        self.parent = parent
        self.state_manager = state_manager
        self.moves = state_manager.get_moves()


        self.num_children = len(moves)

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
        #self.value = 0.0

        # get the children state managers and their vec representations
        state_vecs, state_mans = self.state_manager.moves2vec()

        #get the predicted p and v values for all the children
        p, v = network_wrapper.forward(state_vecs)
        self.v = np.mean(v)

        for i, (p, next_state) in enumerate(zip(p, state_mans)):
            child = Node(state_manager = next_state,
                         parent = self, idx = i)
            #child.idx = i
            #self.v[i] = 0
            self.p[i] = p
            self.w[i] = 0
            self.n[i] = 0
            self.q[i] = 0
            self.u[i] = 0
            self.children.append(child)
        return self

    def update(self, child_idx, v):
        self.n[child_idx] += 1.0
        self.w[child_idx] += v
        self.q[child_idx] = (1 / self.n[child_idx]) * self.w[child_idx]
        #s = #sum(list(map(lambda x: self.n[x], self.children)))
        children_visits = np.sum(self.n)
        c_puct = 1.0
        for i,_ in enumerate(self.children):
            # u will change for all children due to change in the summation of n[child]
            # maybe some actual value for this constant?
            self.u[i] = c_puct * self.p[i] * np.sqrt(s) / (1 + self.n[child_idx])

    def export_pi(self, move_number, temp_change_iter, temp_early, temp_late):
        temperature = temp_early if move_number < temp_change_iter else temp_late
        #return list(map(lambda x: self.n[x] ** (1.0 / temperature), self.children))
        temp_vals = np.power(self.n, 1 / temperature)

        return temp_vals / np.sum(temp_vals)


    '''
    def calc_value(self):
        v = 0.0
        for child in self.children:
            v += self.v[child]
        return v / len(self.children)
    '''
# ============================================================================================================================ #
# MCTS
# ============================================================================================================================ #

class MCTS(object):

    def __init__(self, network_wrapper, temp_change_iter=30, temp_early=1, temp_late = 0.33):
        self.network_wrapper = network_wrapper
        self.root = None
        self.root = Node(state_manager = state_manager, parent = None)
        self.temp_change_iter = temp_change_iter
        self.temp_early = temp_early
        self.temp_late = temp_late

    def __call__(self, state_manager, n = 1500):
        '''
        if not self.root:
            # print('starting mcts simulation ...')
            self.root = Node(state_manager = state_manager, parent = None)
            # print('root is:', self.root)
        else:
            # state_manager.output()
            self.update_root(state_manager)
        '''
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

    '''
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
    '''

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

    def back_propagate(self, node):
        # this is the value predicted for the edge leading to this node by the NN
        #v = node.calc_value()
        while node.parent is not None:
            node.parent.update(child_idx = node.idx, v = self.v)
            node = node.parent
