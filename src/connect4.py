from state_manager import StateManager
import numpy as np

PIECE_MAP = {'x': 0, 'o': 1}

INPUT_SHAPE = (5, 6, 7)

class Connect4:
    def __init__(self, board=None, turn=True):
        self.rows = 6
        self.cols = 7
        self.win_len = 4
        self.empty = ' '

        if board == None:
            self.new_game()
        else:
            self.state = board

        self.turn = turn
        # turn == True ==> white = x
        self.white_moves = 0
        # turn == False ==> black = o
        self.black_moves = 0

    def copy(self):
        return Connect4([row[:] for row in self.state], self.turn)

    def new_game(self):
        self.turn = True
        self.state = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.empty)
            self.state.append(row)

    @property
    def legal_moves(self):
        moves = []
        for i in range(self.cols):
            if self.state[0][i] == self.empty:
                moves.append(i)
        return moves

    def output(self):
        for row in self.state:
            print(row)
        print()

    def __repr__(self):
        s = ""
        for row in self.state:
            s += str(row) + '\n'
        return s

    def apply_move(self, move):
        if self.turn:
            self.white_moves += 1
        else:
            self.black_moves += 1

        for row in reversed(self.state):
            if row[move] == self.empty:
                row[move] = 'x' if self.turn else 'o'
                self.turn = not self.turn
                return True

    def has_winner(self):
        win = lambda l: all(x == 'o' for x in l) or all(x == 'x' for x in l)

        # row win
        for row in self.state:
            for candidate_start in range(self.cols - self.win_len + 1):
                if win(row[candidate_start:candidate_start+self.win_len]):
                    return True

        # col win
        for col in range(self.cols):
            for start in range(self.rows - self.win_len + 1):
                if win([self.state[start + y][col] for y in range(self.win_len)]):
                    return True

        # diag win upper left->lower right
        for i in range(self.rows - self.win_len + 1):
            for j in range(self.cols - self.win_len + 1):
                if win([self.state[i + y][j + y] for y in range(self.win_len)]):
                    return True

        # diag win lower left->upper right
        for i in range(self.win_len - 1, self.rows):
            for j in range(self.cols - self.win_len + 1):
                if win([self.state[i - y][j + y] for y in range(self.win_len)]):
                    return True
        return False


class Connect4Manager(StateManager):
    def __init__(self, state=None):
        super().__init__()

        if state == None:
            self.new_game()
        else:
            self.state = state

    def new_game(self):
        self.state = Connect4()
        self.state.new_game()

    def get_moves(self):
        self.moves = self.state.legal_moves
        return self.moves

    def make_move(self, move_index):
        copy = self.state.copy()

        copy.apply_move(self.moves[move_index])
        print(copy)
        return Connect4Manager(copy)

    def state2vec(self):
        #return [j for i in self.state.state for j in i]
        raise NotImplementedError

    def single_state2vec(self, include_player_pane=False):
        state = self.state

        outvec = np.zeros((2 + include_player_pane, state.rows, state.cols))

        if include_player_pane: outvec[2].fill(not self.turn())

        for i in range(state.rows):
            for j in range(state.cols):
                piece = state.state[i][j]
                if piece != ' ':
                    pane = PIECE_MAP[piece]
                    outvec[pane][i][j] = 1

        return outvec

    def turn(self):
        return self.state.turn

    def __repr__(self):
        return self.state.__repr__()

    def current_state(self):
        return Connect4Manager(self.state.copy())

    def is_terminal_state(self):
        return self.state.has_winner() or self.is_draw()

    def is_win(self):
        return self.state.has_winner()

    def is_draw(self):
        return False

    def zero_is_winner(self):
        return self.state.has_winner() and self.state.turn == False

    def one_is_winner(self):
        return self.state.has_winner() and self.state.turn == True

    def output(self):
        self.state.output()

    def num_full_moves(self):
        return self.white_moves + self.black_moves
