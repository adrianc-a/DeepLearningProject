from state_manager import StateManager

class Connect4:
    def __init__(self, board=None, turn=True):
        self.rows = 6
        self.cols = 7
        self.win_len = 4
        self.empty = ' '
        self.state = board
        self.turn = turn

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

    def apply_move(self, move):
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
        self.state=state

    def new_game(self):
        self.state = Connect4()
        self.state.new_game()

    def get_moves(self):
        self.moves = self.state.legal_moves()
        return self.moves

    def make_move(self, move_index):
        copy = self.state.copy()

        copy.apply_move(self.moves[move_index])
        return Connect4Manager(copy)

    def state2vec(self):
        return [j for i in self.state.state for j in i]

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
