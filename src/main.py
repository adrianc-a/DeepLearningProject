import players
import alphago as ag
import argparse
from game import Game
from game import Evaluator
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager
from chess_manager import ChessManager
from networks import NetworkWrapper, OPTIMIZER_REG
from sys import argv
import tensorflow as tf
import numpy as np

import plotly.plotly as plotly
import plotly.graph_objs as graph_objs

from IPython.display import clear_output, Image, display, HTML

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train/play games with alphago techniques'
    )

    parser.add_argument('-t', '--train-model', action='store_true')
    parser.add_argument('-l', '--load-model', action='store_true')
    parser.add_argument('-m', '--model-name', action='store_true')

    parser.add_argument('-g', '--game', choices=['ttt', 'c4', 'chess'], required=True)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-n', '--num_games', type=int)
    parser.add_argument('-a', '--learning_rate', type=float)
    parser.add_argument('-p', '--play-game', action='store_true')
    parser.add_argument('-q', '--players',
        choices=['alphago', 'human','simple', 'random'], nargs='+')
    parser.add_argument('-s', '--save-model', action='store_true')
    parser.add_argument('-f', '--save-file', type=str)
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'adam'])
    parser.add_argument('-e', '--eval', action='store_true')

    return parser.parse_args(argv[1:])


def train_model(game, iterations, num_games):
    if game == 'ttt':
        player = ag.AlphaGoZeroArchitectures.ttt()
    elif game == 'c4':
        player = ag.AlphaGoZeroArchitectures.connect4_net()
    else:  # chess
        player = ag.AlphaGoZeroArchitectures.chess()

    manager = get_manager(game)

    trainer = ag.AlphaGoZeroTrainer(player)
    trainer.train(manager, iterations=iterations, games=num_games)

    return player


def load_model(game, path, opt):
    if game == 'ttt':
        shape = ag.AlphaGoZeroArchitectures.ttt_input_shape()
    elif game == 'c4':
        shape = ag.AlphaGoZeroArchitectures.connect4_input_shape()
    else:
        shape = ag.AlphaGoZeroArchitectures.chess_input_shape()

    return ag.AlphaGoZero(
        NetworkWrapper.restore(path, shape, opt), get_manager(game)
    )


def get_manager(game):
    if game == 'ttt':
        return TicTacToeManager()
    elif game == 'c4':
        return Connect4Manager()
    elif game == 'chess':
        return ChessManager()


def get_human_player(game):
    if game == 'ttt':
        return players.ttt_human_player
    elif game == 'c4':
        return players.connect_human_player
    elif game == 'chess':
        return players.chess_human_player


def get_players(game, player, ag_player):
    if player == 'alphago':
        return ag_player.play_move
    elif player == 'human':
        return get_human_player(game)
    elif player == 'simple':
        return players.simple_player
    elif player == 'random':
        return players.random_player


def run_model(args):
    if args.train_model:
        ag_player = train_model(args.game, args.iterations, args.num_games)
        ag_player.mcts._begin_game()
    elif args.load_model:
        opt = OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate)
        ag_player = load_model(args.game, args.save_file, opt)
        ag_player.mcts._begin_game()
    else:
        ag_player = None

    if args.play_game or args.eval:
        p1 = get_players(args.game, args.players[0], ag_player)
        p2 = get_players(args.game, args.players[1], ag_player)

    if args.play_game:
        player1_notify=lambda x: None
        player2_notify=lambda x: None

        if args.players[0] == 'alphago':
            player1_notify = ag_player.notify_move
        elif args.players[1] == 'alphago':
            player2_notify = ag_player.notify_move

        Game(get_manager(args.game), p1, p2,
                player1_notify=player1_notify, player2_notify=player2_notify).play()

    if args.eval:
        print(Evaluator(get_manager(args.game), p1, p2,\
            player1_notify = player1_notify, player2_notify = player2_notify,
            player1_name = args.players[0], player2_name = args.players[1],
            should_rate = True, rate_after_each_game = False, evaluation_output = 'ratings').evaluate())

    if args.save_model:
        ag_player.nn.save(args.save_file)

def evaluate_over_time(args):
    player = get_players(args.game, args.players[0], ag_player)
    # 
    model_directory = os.path.asbpath(os.path.join(os.path.dirname(__file__), '../model'))
    # get all checkpoints for this game and model
    checkpoints = []
    for directory in os.listdir(model_directory):
        if os.path.isdir(os.path.join(model_directory, directory)):
            if directory.startswith(args.game + '_' + args.model_name + '_'):
                checkpoints.append(os.path.join(model_directory, directory))
    # 
    evaluation_output_file = args.game + '_' + 'alphago' + '_' + 'checkpoints'
    # sort lexicographically
    checkpoints = sorted(checkpoints)
    notifiers = []
    players = []
    stats = {}
    opt = OPTIMIZER_REG[args.optimizer](learning_rate = args.learning_rate)
    #
    for checkpoint in checkpoints:
        checkpoint_number = checkpoint.split('_')[-1]
        ag_player = load_model(args.game, checkpoint, opt)
        players.append(ag_player)
        notifiers.append(ag_player.notify_move)
        stats.append({
            'e': 0,
            'p': 0,
            'elo': 750,
            'wins': 0,
            'draws': 0,
            'games': 0,
        })
        # 
    num_games = 10
    for iteration in range(args.iterations):
        for i in range(len(checkpoints)):
            for j in range(i + 1, len(checkpoints)):
                iteration_stats = {
                    'r' = [stats[i]['elo'], stats[j]['elo']],
                    'e' = [0, 0],
                    'p' = [0, 0],
                }
                Evaluator(get_manager(args.game), players[i], players[j],
                    player1_notify = notifiers[i],
                    player2_notify = notifiers[j],
                    player1_name = 'checkpoint_' + str(iteration) + '_' + str(i),
                    player2_name = 'checkpoint_' + str(iteration) + '_' + str(j),
                    should_rate = True, rate_after_each_game = False, should_import_export_ratings = False,
                    game_stats = iteration_stats, update_ratings = update_ratings).evaluate(num_games = num_games)
                stats[i]['p'] += iteration_stats[0]['p']
                stats[i]['e'] += iteration_stats[0]['e']
                stats[i]['p'] += iteration_stats[1]['p']
                stats[i]['e'] += iteration_stats[1]['e']
        for player in range(len(players)):
            update_player_ratings(num_games, stats, player)
    # 
    plot_elo_ratings(args, stats, evaluation_output_file)

def update_player_ratings(num_games, stats, player):
    stats[player]['games'] += num_games
    # effective number of games
    n_star = int(50.0 / ((0.662 + 0.00000739 * ((2569 - stats[player]['elo']) ** 2)) ** 0.5))\
        if stats[player]['elo'] <= 2355 else 50.0
    # 
    effective_n = min(stats[player]['games'], n_star)
    #
    k = 800.0 / (effective_n + num_games)
    stats[player_name]['elo'] = stats[player][player_index] + k * (stats[player]['p'] - stats[player]['e'])

def plot_elo_ratings(args, stats, evaluation_output_file):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + evaluation_output_file + '.json'))
    with open(path, 'r') as json_file:
        ratings = json.load(json_file)
    evaluation_file = args.game + '_' + 'alphago' + '_' + 'checkpoints'
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + evaluation_output_file + '.html'))
    data = []
    for player in ratings:
        print(player, ':', ratings[player]['elo'])
        data.append(ratings[player]['elo'])
        # Create traces
    plotly.offline.plot(data, filename = path)
    
def evaluate_against_each_other(args):
    if args.train_model:
        ag_player = train_model(args.game, args.iterations, args.num_games)
    elif args.load_model:
        opt = OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate)
        ag_player = load_model(args.game, args.save_file, opt)
    else:
        ag_player = None
    p1 = get_players(args.game, args.players[0], ag_player)
    p2 = get_players(args.game, args.players[1], ag_player)
    # 
    player1_notify = lambda x: None
    player2_notify = lambda x: None
    if args.players[0] == 'alphago':
        player1_notify = ag_player.notify_move
    elif args.players[1] == 'alphago':
        player2_notify = ag_player.notify_move
    # 
    evaluation_file = args.players[0] + '_vs_' + args.players[1]
    # 
    Evaluator(get_manager(args.game), p1, p2,
        player1_notify = player1_notify, player2_notify = player2_notify,
        player1_name = args.game + '_' + args.model_name + '_' + checkpoint_number, player2_name = args.players[0],
        should_rate = True, rate_after_each_game = False, evaluation_output = evaluation_file).evaluate()

if __name__ == '__main__':
    run_model(parse_args())
    # evaluate_against_each_other(parse_args())
