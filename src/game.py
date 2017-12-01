from enum import Enum

class Evaluator:
    def __init__(self, manager, player1, player2):
        self.game = Game(
            manager, player1, player2, end_game=self._end_game, log=False,
            render=False
        )
        self.player1_wins = 1
        self.player2_wins = 1

    def _end_game(self, res, winner):
        if res == GameResult.WIN:
            if winner == 0:
                self.player1_wins += 1
            else:
                self.player2_wins += 1
        elif res == GameResult.DRAW:
            self.player1_wins += .1
            self.player2_wins += .1



    def evaluate(self, num_games=5):
        self.game.play(num_games)

        player1_to_2 = self.player1_wins / (self.player2_wins + self.player1_wins)

        if player1_to_2 >= 0.55:
            return 0, self.game.player1
        else:
            return 1, self.game.player2


class GameResult(Enum):
    WIN  = 0
    DRAW = 1
    # I have this as some catch-all.
    # Can we guarantee, that a game is either won or drawn?
    # will is_draw || is_win always be true when we have a end state?
    # I know for ttt and c4 it is true. what about chess?
    END  = 2

class Game:

    def __init__(self, manager, player1, player2, begin_game=lambda: None,
            end_game=lambda t, w: None, log=True, render=True):
        self.manager = manager
        self.player1 = player1
        self.player2 = player2
        self.end_game = end_game
        self.begin_game = begin_game
        self.log = True
        self.render = render

    def play(self, num_games=1):
        for i in range(num_games):
            self.begin_game()
            self.end_game(*self._play_game())

    def _play_game(self):
        self.manager.new_game()
        if self.log:
            self.manager.output()

        # true means player1's turn, False player2
        turn = True

        n = 0
        move_idx = -1
        while not self.manager.is_terminal_state():
            moves = self.manager.next_states()

            if len(moves) == 0:
                print('draw')
                return

            if self.log:
                print('turn: ', turn)

            player = self.player1 if turn else self.player2

            move_idx = player(self.manager.current_state(), moves, move_idx)

            if move_idx < 0 or move_idx >= len(moves):
                print('illegal move made')
                return

            self.manager = self.manager.make_move(move_idx)
            if self.log:
                self.manager.output()
            if self.render:
                self.manager.render(n)
            n += 1
            turn = not turn
            if self.log:
                print('\n==================================================\n')
        if self.manager.is_win():
            winner = 0 if self.manager.zero_is_winner() else 1
            if self.log:
                print(
                    'The winner is: ' +
                        ('player1' if winner == 0 else 'player2')
                )
            return GameResult.WIN, winner
        elif self.manager.is_draw():
            if self.log:
                print('Draw')
            return GameResult.DRAW, 0
        else:
            if self.log:
                print('Game ended')
            return GameResult.END, 0

