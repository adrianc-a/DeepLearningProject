import players


class Game:
    def __init__(self, player1, player2, *output_args):
        self.turn = True
        self.player1 = player1
        self.player2 = player2

    def new_game(self):
        pass
    # question: how to give the player function access to this?
    def legal_moves(self):
        pass

    def has_winner(self):
        pass

    def apply_move(self, move):
        pass

    def output(self):
        pass

    def play(self):
        self.new_game()
        self.output()

        while not self.has_winner():
            moves = self.legal_moves()

            if len(moves) == 0:
                print('draw?')
                return

            player = self.player1 if self.turn else self.player2
            move = player(self.state)

            if move not in self.legal_moves():
                print('illegal move made')
                return
            self.apply_move(move)
            self.output()

            self.turn = not self.turn

        return self.turn


class TTT(Game):
    def new_game(self):
        self.state = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

    def legal_moves(self):
        moves = []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def output(self):
        for r in self.state:
            print(r)
        print()

    def has_winner(self):
        win = lambda l: all(x == 'o' for x in l) or all(x == 'x' for x in l)

        for r in range(3):
            # row win
            if win(self.state[r]): return True

            # col win
            if win([self.state[c][r] for c in range(3)]): return True

        # diagonal wins
        if win([self.state[i][i] for i in range(3)]): return True
        if win([self.state[2 - i][i] for i in range(3)]): return True

        return False

    def apply_move(self, move):
        self.state[move[0]][move[1]] = 'x' if self.turn else 'o'

    def play(self):
        winner = super().play()

        print('winner is ' + ('o' if winner else 'x'))


class Chess(Game):
    pass


# specifically for the AlphaGo Zero Player
class GameTree:
    # maintain game tree, specific to game
    def __init__(self, nn):
        self.nn = nn
        self.edges = []
        self.children = []
        self.value = None

    # updates the tree
    def mcts(self, simulations):
        pass

    def make_move(self, state):
        pass

    # feel free to change the names
    def backup(self):
        pass

    def forward(self):
        pass


# question: is this necessary?
class Player:
    def make_move(self, state, who):
        raise NotImplementedError()

