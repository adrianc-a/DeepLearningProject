import chess
from state_manager import StateManager
import numpy as np

DEFAULT_BOARD = chess.Board()

class ChessManager(StateManager):

    def __init__(self, board=DEFAULT_BOARD):
        super().__init__()
        self.new_game(board)

    def new_game(self, board=DEFAULT_BOARD):
        self.board = board.copy()
        self.current_legal_moves = []

    def get_moves(self):
        """
        Returns a list of available moves in a list, to play a particular one
        pass its index to the make_move function.
        """
        self.current_legal_moves = list(self.board.legal_moves)
        return self.current_legal_moves

    def make_move(self, move_index):
        next_state = self.board.copy()
        uci = self.current_legal_moves[move_index]

        next_state.push(uci)
        return ChessManager(next_state)

    def state2vec(self):
        """ Outputs a 3-d tensor of the board state

        NOTE: the board orientation is reversed relative to what
        the printed reprentation is (i.e. white starts at the top left of the array)
        """

        outvec = np.zeros((3,8,8))

        pieces = list(self.board.piece_map().items())
        whites = ((idx,piece.piece_type) for idx, piece in pieces if piece.color == chess.WHITE)
        blacks = ((idx,piece.piece_type) for idx, piece in pieces if piece.color == chess.BLACK)

        white_pane = outvec[0].reshape((64,))
        black_pane = outvec[1].reshape((64,))

        for idx, p in whites: white_pane[idx] = p
        for idx, p in blacks: black_pane[idx] = p

        outvec[2].fill(self.turn())

        return outvec


    #presumabely different from checking if a checkmate occured
    def is_terminal_state(self):
        return self.board.is_game_over()

    def is_win(self):
        """
        Returns if a checkmate has occured, this should be called before
        checking who the winner is.
        """
        return self.board.is_checkmate()

    def turn(self):
        """
        Returns a 0 if it is the 1st players turn and a 1 if it is the
        second player's turn.
        """
        # maybe just want to return the boolean?
        # also False == Black, True == White hence the not
        return int(not self.board.turn)

    def current_state(self):
        return ChessManager(self.board.copy())

    def is_draw(self):
        return self.board.is_stalemate()

    def zero_is_winner(self):
        """Checks that player 0 (white) is the winner"""
        return self.is_checkmate() and self.turn() == 1

    def one_is_winner(self):
        """Checks that player 1 (black) is the winner"""
        return self.is_checkmate() and self.turn() == 0

    def num_full_moves(self):
        """Returns number of full moves played
           (i.e. for each pair of moves by white and black)"""
        return int(self.fen()[-1])

    # allays the need for call to output()
    def __str__(self):
        return self.board.__str__()

    def output(self):
        print(self.board)
