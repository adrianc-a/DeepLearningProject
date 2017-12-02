from enum import Enum

import os
import json
import math

class Evaluator:
    def __init__(self, manager, player1, player2,
            player1_notify=lambda x: None, player2_notify=lambda x: None, begin_game=lambda: None,
            player1_name = 'player 1', player2_name = 'player 2', should_rate = False,
            rate_after_each_game = False, evaluation_output = 'ratings'):
        self.game = Game(
            manager, player1, player2, end_game=self._end_game, log=False,
            render=False,
            player1_notify=player1_notify,
            player2_notify=player2_notify,
            begin_game=begin_game
        )
        self.should_rate = should_rate
        self.rate_after_each_game = rate_after_each_game
        self.evaluation_output = evaluation_output
        self.player1_wins = 1
        self.player2_wins = 1
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.elo_e = [0, 0]
        self.elo_p = [0, 0]
        print('evaluating', self.player1_name, 'vs', self.player2_name)

    def _end_game(self, res, winner):
        if self.should_rate:
            self.elo_e[0] += 1.0 / (1.0 + 10.0 ** ((self.r[1] - self.r[0]) / 400.0))
            self.elo_e[1] += 1.0 / (1.0 + 10.0 ** ((self.r[0] - self.r[1]) / 400.0))
        if res == GameResult.WIN:
            if winner == 0:
                self.player1_wins += 1
                if self.should_rate:
                    self.elo_p[0] += 1.0
                    self.ratings[self.player1_name]['wins'] += 1
            else:
                self.player2_wins += 1
                if self.should_rate:
                    self.elo_p[1] += 1.0
                    self.ratings[self.player2_name]['wins'] += 1
        elif res == GameResult.DRAW:
            self.player1_wins += .1
            self.player2_wins += .1
            if self.should_rate:
                self.elo_p[0] += 0.5
                self.elo_p[1] += 0.5
                self.ratings[self.player1_name]['draws'] += 1
                self.ratings[self.player2_name]['draws'] += 1
        if self.should_rate and self.rate_after_each_game:
            # if yout want to update ratings after each game
            self.update_ratings(1, self.player1_name, 0)
            self.update_ratings(1, self.player2_name, 1)

    def evaluate(self, num_games=5):
        if self.should_rate:
            self.import_ratings()
        self.game.play(num_games)
        if self.should_rate:
            # if yout want to update ratings after each game
            if not self.rate_after_each_game:
                self.update_ratings(num_games, self.player1_name, 0)
                self.update_ratings(num_games, self.player2_name, 1)
            print('exporting ratings')
            self.export_ratings()
        #
        player1_to_2 = self.player1_wins / (self.player2_wins + self.player1_wins)

        if player1_to_2 >= 0.55:
            return 0, self.game.player1
        else:
            return 1, self.game.player2
        #

    def update_ratings(self, num_games, player_name, player_index):
        print('update')
        # 
        self.ratings[player_name]['games'] += num_games
        # effective number of games
        n_star = int(50.0 / ((0.662 + 0.00000739 * ((2569 - self.r[player_index]) ** 2)) ** 0.5))\
            if self.r[player_index] <= 2355 else 50.0
        # 
        effective_n = min(self.ratings[player_name]['games'], n_star)
        #
        k = 800.0 / (effective_n + num_games)
        self.ratings[player_name]['elo'] = self.r[player_index] + k * (self.elo_p[player_index] - self.elo_e[player_index])
        # 
        self.elo_e[player_index] = 0

    def import_ratings(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/'))
        if not os.path.exists(path):
            os.makedirs(path)
        # create the ratings file with a valid empty json structure
        path = os.path.join(path, self.evaluation_output + '.json')
        if not os.path.isfile(path):
            with open(path, 'w+') as json_file:
                json.dump({}, json_file)
        # import the ratings
        with open(path, 'r') as json_file:
            self.ratings = json.load(json_file)
        self.add_player_to_ratings(self.player1_name)
        self.add_player_to_ratings(self.player2_name)
        # to make it more concise
        self.r = [self.ratings[self.player1_name]['elo'], self.ratings[self.player2_name]['elo']]

    def add_player_to_ratings(self, player):
        if not player in self.ratings:
            self.ratings[player] = {
                'elo': 750,
                'wins':  0,
                'games': 0,
                'draws': 0,
            }

    def export_ratings(self):
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
        self.log = log
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

