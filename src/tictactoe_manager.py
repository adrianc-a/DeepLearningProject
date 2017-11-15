import tictactoe as ttt
from state_manager import StateManager

DEFAULT_BOARD = ttt.TTT()

class TicTacToeManager(StateManager):

    def __init__(self, board=DEFAULT_BOARD):
        super().__init__()
        self.board = board.copy()
        self.current_legal_moves = []

    def get_moves(self):
        self.current_legal_moves = list(self.board.legal_moves)
        return self.current_legal_moves

    def make_move(self, move_index):
        next_state = self.board.copy()
        idx = self.current_legal_moves[move_index]
        
        next_state.push(idx)
        return TicTacToeManager(next_state)
    
    def is_terminal_state(self):
        return self.is_three_in_a_row() or self.board.draw()

    def is_draw(self):
        return self.board.draw()

    def zero_is_winner(self):
        return self.board.tiar(0)

    def one_is_winner(self):
        return self.board.tiar(1) 

    def turn(self):
        return self.board.turn

    def num_full_moves(self):
        return self.num_full_moves()
