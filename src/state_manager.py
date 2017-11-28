import numpy as np

class StateManager:
    """
    Abstract base class to wrap around actual implementations of games
    e.g. chess/tic-tac-toe abstracts logic to get available moves from
    a given state, make a move, etc.

    """

    def __init__(self):
        pass

    def new_game(self):
        pass

    def get_moves(self):
        pass

    def make_move(self, move_index):
        pass

    def state2vec(self):
        pass

    def next_states(self):
        return [self.make_move(i) for i,_ in enumerate(self.get_moves())]

    '''
    def moves2vec(self):
        "Returns a batch of state tensors for input to the network"
        return np.concatenate([state.state2vec() for state in self.next_states()])
    '''

    def moves2vec(self):
        cur_state_vec = self.single_state2vec(include_player_pane=True)

        next_states = self.next_states()
        next_vecs = [next_state.single_state2vec() for next_state in next_states]

        return np.stack(
                 [np.concatenate([next_vec,cur_state_vec])
                 for next_vec in next_vecs]), next_states()

    # each subclass should return a new instance of itself, with the current
    # board state, or a copy
    def current_state(self):
        pass

    def is_terminal_state(self):
        pass

    def is_win(self):
        pass

    def is_draw(self):
        pass

    def zero_is_winner(self):
        pass

    def one_is_winner(self):
        pass

    def output(self):
        pass

    def render(self, n):
        pass

    # number of moves (white & black)
    def num_full_moves(self):
        pass
