import tictactoe as ttt
from state_manager import StateManager
import numpy as np

DEFAULT_BOARD = ttt.TTT()

TILE_MAP = {'X': 0, 'O': 1}

INPUT_SHAPE = (5,3,3)

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

    def single_state2vec(self, include_player_pane=False):
        """Returns the board state represented as a Tensor

        Args:
            include_player_pane (Boolean): set to true to include a pane
            where all entries are set to the numeric value of the current
            player this pane is appended as the final pane of the output

        Returns the state of the tic-tac-toe game as 3-d tensor

        pane 1 has a 1 if player 0 has a mark in that particular location
        pane 2 has a 1 if player 1 has a mark in that particular location
        pane 3 is either all 0 or all 1 depending on whose turn it is
        """

        # one channel for x's one for o's and one
        # to denote the whose turn it is
        outvec = np.zeros((2 + include_player_pane,3,3))

        if include_player_pane: outvec[2].fill(self.turn())

        for i in range(3):
            for j in range(3):
                tile = self.board.board[i * 3 + j]
                if tile == '.':
                    continue
                else:
                    pane_num = TILE_MAP[tile]
                    outvec[pane_num][i][j] = 1

        return outvec

    def state2vec(self, for_next=False):
        """DEPRECATED: Returns the board state represented as a Tensor

        Args:
            for_next (Boolean): set to true if you're generating moves for the
            next set of states (makes sure that the color pane of the output
            vector is properly set)

        Returns the state of the tic-tac-toe game as 3-d tensor

        pane 1 has a 1 if player 0 has a mark in that particular location
        pane 2 has a 1 if player 1 has a mark in that particular location
        pane 3 is either all 0 or all 1 depending on whose turn it is
        """
        '''
        # one channel for x's one for o's and one
        # to denote the whose turn it is
        outvec = np.zeros((3,3,3))

        # fill in whose turn it was
        outvec[2].fill( (self.turn() + for_next) % 2)

        for i in range(3):
            for j in range(3):
                tile = self.board.board[i * 3 + j]
                if tile == '.':
                    continue
                else:
                    pane_num = TILE_MAP[tile]
                    outvec[pane_num][i][j] = 1

        return outvec.reshape((1,3,3,3))
        '''
        raise NotImplementedError

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

    @staticmethod
    def cl_name():
        return 'ttt'

    def name(self):
        return TicTacToeManager.cl_name()

def state2vec_singledim(self):
        piece_map = {'.': 0, 'X': 1, 'O':2}

        # one extra element in vector to denote whose turn
        xs = np.zeros((10,))
        for i in range(9):
            xs[i] = piece_map[self.board.board[i]]

        xs[9] = self.board.turn #0 for player 0, 1 for player 2

        return xs
