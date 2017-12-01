from enum import Enum

import os
import json

class Evaluator:
    def __init__(self, manager, player1, player2,
        player1_name = 'player 1', player2_name = 'player 2', rate_players = False, evaluation_output = 'ratings'):
        self.game = Game(
            manager, player1, player2, end_game=self._end_game, log=False,
            render=False
        )
        self.should_rate = rate_players
        self.evaluation_output = evaluation_output
        self.player1_wins = 1
        self.player2_wins = 1

    def _end_game(self, res, winner):
        e = [0, 0]
        p = [0, 0]
        e[0] = 1.0 / (1.0 + 10.0 ** ((self.ratings[player2_name]['elo'] - self.ratings[player1_name]['elo']) / 400.0))
        e[1] = 1.0 / (1.0 + 10.0 ** ((self.ratings[player1_name]['elo'] - self.ratings[player2_name]['elo']) / 400.0))
        if res == GameResult.WIN:
            if winner == 0:
                self.player1_wins += 1
                if should_rate:
                    p[0] = self.ratings[1] + 400.0
                    p[1] = self.ratings[0] - 400.0
                    self.ratings[player_1]['wins'] += 1
            else:
                self.player2_wins += 1
                if should_rate:
                    p[0] = self.ratings[1] - 400.0
                    p[1] = self.ratings[0] + 400.0
                    self.ratings[player_2]['wins'] += 1
        elif res == GameResult.DRAW:
            self.player1_wins += .1
            self.player2_wins += .1
            if should_rate:
                self.ratings[player_1]['draws'] += 1
                self.ratings[player_2]['draws'] += 1
        if should_rate:
            self.ratings[player_1]['games'] += 1
            self.ratings[player_2]['games'] += 1
            k = 800.0 / (self.ratings[player1_name]['games'])
            self.ratings[player1_name]['elo'] = self.ratings[player1_name]['elo'] + k * (p[0] - e[0])
            k = 800.0 / (self.ratings[player2_name]['games'])
            self.ratings[player2_name]['elo'] = self.ratings[player2_name]['elo'] + k * (p[1] - e[1])

    def evaluate(self, num_games=5):
        self.import_ratings()
        self.game.play(num_games)
        #
        player1_to_2 = self.player1_wins / (self.player2_wins + self.player1_wins)

        if player1_to_2 >= 0.55:
            return 0, self.game.player1
        else:
            return 1, self.game.player2
        #
        self.export_ratings()

    def import_ratings(self):
        if self.should_rate:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + self.evaluation_output + '.json'))
            # create the ratings file with a valid empty json structure
            if not os.path.isfile(path):
                with open(path, 'w+') as json_file:
                    json.dump({}, json_file)
            # import the ratings
            with open(path, 'r') as json_file:
                self.ratings = json.load(json_file)
            add_player_to_ratings(player1_name)
            add_player_to_ratings(player2_name)

    def add_player_to_ratings(self, player):
        if not player in self.ratings:
            self.ratings[player] = {
                'elo': 1300,
                'wins':  0,
                'games': 0,
                'draws': 0,
            }

    def export_ratings(self):
        if self.should_rate:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + self.evaluation_output + '.json'))
            with open(path, 'w') as json_file:
                json.dump(self.ratings, json_file)

class GameResult(Enum):
    WIN  = 0
    DRAW = 1
    # I have this as some catch-all.
    # Can we guarantee, that a game is either won or drawn?
    # will is_draw || is_win always be true when we have a end state?
    # I know for ttt and c4 it is true. what about chess?
    END  = 2

class Game:

    def __init__(self, manager, player1, player2, begin_play=lambda: None, begin_game=lambda: None,
            end_game=lambda t, w: None, log=True, render=True,
            player1_notify=lambda i: None, player2_notify = lambda i: None):
        self.manager = manager
        self.player1 = player1
        self.player2 = player2
        self.begin_play = begin_play
        self.end_game = end_game
        self.begin_game = begin_game
        self.log = True
        self.render = render
        self.player1_notify = player1_notify
        self.player2_notify = player2_notify

    def play(self, num_games=1):
        self.begin_play()
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
        while not self.manager.is_terminal_state():
            moves = self.manager.next_states()

            if len(moves) == 0:
                print('draw')
                return

            if self.log:
                print('turn: ', turn)

            player = self.player1 if turn else self.player2

            move_idx = player(self.manager.current_state(), moves)

            if move_idx < 0 or move_idx >= len(moves):
                print('illegal move made')
                return

            self.manager = self.manager.make_move(move_idx)

            player_notifier = self.player1_notify if turn else self.player2_notify

            player_notifier(move_idx)

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

