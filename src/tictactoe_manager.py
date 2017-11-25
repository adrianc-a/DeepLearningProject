import tictactoe as ttt
from state_manager import StateManager
import numpy as np

DEFAULT_BOARD = ttt.TTT()

TILE_MAP = {'X': 0, 'O': 1}

class TicTacToeManager(StateManager):

    def __init__(self, board=DEFAULT_BOARD):
        super().__init__()
        self.new_game(board)

    def new_game(self, board=DEFAULT_BOARD):
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

    def state2vec(self):
        """Returns the board state represented as a Tensor

        Returns the state of the tic-tac-toe game as 3-d tensor

        pane 1 has a 1 if player 0 has a mark in that particular location
        pane 2 has a 1 if player 1 has a mark in that particular location
        pane 3 is either all 0 or all 1 depending on whose turn it is
        """

        # one channel for x's one for o's and one
        # to denote the whose turn it is
        outvec = np.zeros((3,3,3))

        # fill in whose turn it was
        outvec[2].fill( (self.turn() + 1) % 2)


        for i in range(3):
            for j in range(3):
                tile = self.board.board[i * 3 + j]
                if tile == '.':
                    continue
                else:
                    pane_num = TILE_MAP[tile]
                    outvec[pane_num][i][j] = 1

        return outvec.reshape((1,3,3,3))


    def current_state(self):
        return TicTacToeManager(self.board.copy())

    def is_terminal_state(self):
        return self.board.is_three_in_a_row() or self.board.draw()

    def is_draw(self):
        return self.board.draw()

    def zero_is_winner(self):
        return self.board.tiar(0)

    def one_is_winner(self):
        return self.board.tiar(1)

    def is_win(self):
        return self.board.is_three_in_a_row()

    def is_draw(self):
        return self.board.draw()

    def turn(self):
        return self.board.turn

    def num_full_moves(self):
        return self.board.move_count

    def output(self):
        print(self.board)

def state2vec_singledim(self):
        piece_map = {'.': 0, 'X': 1, 'O':2}

        # one extra element in vector to denote whose turn
        xs = np.zeros((10,))
        for i in range(9):
            xs[i] = piece_map[self.board.board[i]]

        xs[9] = self.board.turn #0 for player 0, 1 for player 2

        return xs
