

class StateManager():
    """
    Abstract base class to wrap around actual implementations of games
    e.g. chess/tic-tac-toe abstracts logic to get available moves from
    a given state, make a move, etc.

    """

    def __init__(self):
        pass 

    def get_moves(self):
        pass

    def make_move(self, move_index):
        pass

    def moves2vec(self):
        pass

    def move2vec(self):
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

