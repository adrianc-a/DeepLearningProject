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
        states = []
        for i in range(len(self.get_moves())):
            states.append(self.make_move(i))
        return states

    def moves2vec(self):
        v_moves = []
        for state in self.next_states():
            v_moves.append(state.state2vec())
        return v_moves

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
