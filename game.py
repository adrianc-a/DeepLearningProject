import players


class Tree():
    # maintain edges, vertices etc.
    pass


class Game():
    def __init__(self, player1, player2):
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

    def play(self):
        self.new_game()

        while not self.has_winner():
            moves = self.legal_moves()

            if len(moves) == 0:
                print('draw?')
                return

            if self.turn:
                player = self.player1
            else:
                player = self.player2
            move = player(self.state)

            if move not in self.legal_moves():
                print('illegal move made')
                return
            self.apply_move(move)

            self.turn = not self.turn

        print('winner is ' + str(self.turn))


class TTT(Game):
    pass


class Chess(Game):
    pass


# specifically for the AlphaGo Zero Player
class GameTree():
    # maintain game tree, specific to game
    def __init__(self, state_manager, nn):
        self.tree = None

    # updates the tree
    def mcts(self, simulations):
        pass

    def play_move(self):
        pass

    # feel free to change the names
    def backup(self):
        pass

    def forward(self):
        pass


# question: is this necessary?
class Player():
    def make_move(self, state, who):
        raise NotImplementedError()


player = players.ttt_alphago_zero_player()

player.train(num_iters=20)

g = TTT(player.make_move, players.ttt_human_player)
g.play()
