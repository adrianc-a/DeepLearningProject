import os
import json

import players as game_players
import alphago as ag
import argparse
from game import Game
from game import Evaluator
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager
from chess_manager import ChessManager
from networks import NetworkWrapper, OPTIMIZER_REG
from sys import argv
from glob import glob

GAMES = ['ttt', 'c4', 'chess']
PLAYERS = ['alphago', 'human', 'simple', 'random']

import plotly.graph_objs as graph_objs
import plotly.offline as plotly

from IPython.display import clear_output, Image, display, HTML

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train/play games with alphago techniques'
    )

    parser.add_argument('-t', '--train-model', action='store_true')
    parser.add_argument('-l', '--load-model', action='store_true')
    parser.add_argument('-g', '--game', choices=GAMES, required=True)
    parser.add_argument('-I', '--iterations', type=int)
    parser.add_argument('-N', '--num-games', type=int)
    parser.add_argument('-A', '--learning-rate', type=float)
    parser.add_argument('-p', '--play-game', action='store_true')
    parser.add_argument('-q', '--players', choices=PLAYERS, nargs='+')
    parser.add_argument('-s', '--save-model', action='store_true')
    parser.add_argument('-f', '--save-file', type=str)
    parser.add_argument('-o', '--optimizer', choices=list(OPTIMIZER_REG.keys()))
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('--elo', action='store_true')
    parser.add_argument('-M', '--momentum', type=float, default=0.01)
    parser.add_argument('-u', '--use-nesterov', action='store_true')
    parser.add_argument('-C', '--checkpoint', type=int, default=15)
    parser.add_argument('-n', '--name', type=str, default='')
    parser.add_argument('-S', '--sample-states', type=int, default=2048)
    parser.add_argument('-E', '--epochs', type=int, default=5)
    parser.add_argument('-B', '--batch-size', type=int, default=32)
    parser.add_argument('-X', '--mcts-searches', type=int, default=10)
    parser.add_argument('-T', '--temp-change-iter', type=int, default=7)
    parser.add_argument('-T1', '--temp-early', type=float, default=1)
    parser.add_argument('-Tn', '--temp-late', type=float, default=0.5)
    parser.add_argument('-V', '--save-point', type=int, default=int(1e20))
    parser.add_argument('-R', '--regularization', type=float, default=0.001)
    parser.add_argument('-z', '--cutoff', action='store_true')
    parser.add_argument('-L', '--max-length', type=int, default=100)

    ret = parser.parse_args(argv[1:])
    print('Running with args:')
    print(ret)
    return ret


def train_model(player, args):
    manager = get_manager(args)

    trainer = ag.AlphaGoZeroTrainer(player, args)
    trainer.train(manager, iterations=args.iterations, games=args.num_games, ckpt=args.checkpoint,
                  savept=args.save_point)

    return player


def load_model(args):
    return ag.AlphaGoZero(
        NetworkWrapper.restore(args.save_file), get_manager(args), args
    )


def get_manager(args):
    if args.game == 'ttt':
        return TicTacToeManager()
    elif args.game == 'c4':
        return Connect4Manager()
    elif args.game == 'chess':
        return ChessManager()


def get_human_player(game):
    if game == 'ttt':
        return game_players.ttt_human_player
    elif game == 'c4':
        return game_players.connect_human_player
    elif game == 'chess':
        return game_players.chess_human_player


def get_players(game, player, ag_player):
    if player == 'alphago':
        return ag_player.play_move
    elif player == 'human':
        return get_human_player(game)
    elif player == 'simple':
        return game_players.simple_player
    elif player == 'random':
        return game_players.random_player

def build_opt(args):
    if args.optimizer == 'sgd' or args.optimizer == 'adam':
        return OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate)
    elif args.optimizer == 'momentum':
        return OPTIMIZER_REG[args.optimizer](learing_rate=args.learning_rate, momentum=args.momentum,
                                             use_nesterov=args.use_nesterov)
    elif args.optimizer == 'rms':
        return OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate, momentum=args.momentum)


def build_player(args):
    opt = build_opt(args)
    if args.game == 'ttt':
        return ag.AlphaGoZeroArchitectures.ttt(opt, args)
    elif args.game == 'c4':
        return ag.AlphaGoZeroArchitectures.connect4_net(opt, args)
    else:  # chess
        return ag.AlphaGoZeroArchitectures.chess_net(opt, args)

def run_model(args):
    ag_player = None
    if args.load_model:
        ag_player = load_model(args)
    if args.train_model:
        if glob('{}_{}*'.format(args.game, args.name)):
            args.name = input('There are files matching this name. Pick another\n')
        if ag_player is None:
            ag_player = build_player(args)
        ag_player = train_model(ag_player, args)

    if args.play_game or args.eval:
        p1 = get_players(args.game, args.players[0], ag_player)
        p2 = get_players(args.game, args.players[1], ag_player)

    if args.play_game:
        player1_notify = lambda x: None
        player2_notify = lambda x: None

        if args.players[0] == 'alphago':
            player1_notify = ag_player.notify_move
        elif args.players[1] == 'alphago':
            player2_notify = ag_player.notify_move

        Game(get_manager(args), p1, p2,
             player1_notify=player1_notify, player2_notify=player2_notify, max_length=args.max_length,
             cutoff=args.cutoff).play()

    if args.eval:
        print(Evaluator(get_manager(args.game), p1, p2,\
            player1_notify = player1_notify, player2_notify = player2_notify,
            player1_name = args.players[0], player2_name = args.players[1],
            should_rate = True, rate_after_each_game = False, evaluation_output = 'ratings').evaluate())

    if args.save_model:
        ag_player.nn.save(args.save_file)

# ============================================================================================================================ #
# Elo scoring and evaluation stuff
# ============================================================================================================================ #

def extract_checkpoints(args):
    model_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    # get all checkpoints for this game and model
    checkpoints = []
    for directory in os.listdir(model_directory):
        if os.path.isdir(os.path.join(model_directory, directory)):
            if directory.startswith(args.game + '_' + args.name + '_'):
                checkpoints.append(os.path.join(model_directory, directory))
    return checkpoints

def evaluate_over_time(args, freeze_previous_ratings = False):
    checkpoints = extract_checkpoints(args)
    # 
    evaluation_output_file = args.game + '_' + 'alphago' + '_' + 'checkpoints'
    # 
    simple_player = game_players.simple_player
    random_player = game_players.random_player
    # sort lexicographically
    checkpoints = sorted(checkpoints)
    notifiers = []
    players = []
    stats = []
    # 
    notifiers.append(lambda x: None)
    players.append(simple_player)
    stats.append({
        'e': 0,
        'p': 0,
        'elo': 750,# + 100 * (len(players) - 1),
        'wins': 0,
        'draws': 0,
        'games': 0,
    })
    # 
    notifiers.append(lambda x: None)
    players.append(random_player)
    stats.append({
        'e': 0,
        'p': 0,
        'elo': 750,# + 100 * (len(players) - 1),
        'wins': 0,
        'draws': 0,
        'games': 0,
    })
    #
    for checkpoint in checkpoints:
        checkpoint_number = checkpoint.split('_')[-1]
        print('found checkpoint ', checkpoint, 'number: ', checkpoint_number)
        args.save_file = checkpoint
        print(args)
        ag_player = load_model(args)
        players.append(ag_player.play_move)
        notifiers.append(ag_player.notify_move)
        stats.append({
            'e': 0,
            'p': 0,
            'elo': 750,# + 100 * (len(players) - 1),
            'wins': 0,
            'draws': 0,
            'games': 0,
        })
        # 
    num_games = 5
    iterations = 1 if freeze_previous_ratings else args.iterations
    for iteration in range(iterations):
        print('iteration:', iteration)
        for i in range(1, len(players)): # start from random player and rate
            stats[i]['elo'] = stats[i - 1]['elo']
            r = range(0, i) if freeze_previous_ratings else range(0, len(players))
            for j in r:
                if i == j: # not playing against ourselves, done enough of that already :D
                    continue
                print('playing', i, ' against', j)
                player1_name = 'player' + '_' + str(i)
                player2_name = 'player' + '_' + str(j)
                # keep e and p separate from other stats as these are ephemeral
                iteration_stats = {
                    player1_name: stats[i],
                    player2_name: stats[j],
                    'e': {player1_name: 0, player2_name: 0},
                    'p': {player1_name: 0, player2_name: 0},
                }
                Evaluator(get_manager(args), players[i], players[j],
                    player1_notify = notifiers[i],
                    player2_notify = notifiers[j],
                    player1_name = player1_name,
                    player2_name = player2_name,
                    should_rate = True, rate_after_each_game = False, should_import_export_ratings = False,
                    game_stats = iteration_stats).evaluate(num_games = num_games)
                # only update the rating for player i, freeze previous players
                stats[i]['p'] += iteration_stats['p'][player1_name]
                stats[i]['e'] += iteration_stats['e'][player1_name]
                if not freeze_previous_ratings:
                    stats[j]['p'] += iteration_stats['p'][player2_name]
                    stats[j]['e'] += iteration_stats['e'][player2_name]
            if freeze_previous_ratings:
                print(i, stats[i])
                update_player_ratings(num_games, stats, i)
        if not freeze_previous_ratings:
            for player in range(len(players)):
                print(stats[player])
                update_player_ratings(num_games, stats, player)
    # export ratings
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + evaluation_output_file + '.json'))
    with open(path, 'w') as json_file:
        json.dump(stats, json_file, sort_keys = True, indent = 4, separators = (',', ': '))
    # 
    plot_elo_ratings(args, evaluation_output_file)

def update_player_ratings(num_games, stats, player):
    stats[player]['games'] += num_games
    # effective number of games
    n_star = int(50.0 / ((0.662 + 0.00000739 * ((2569 - stats[player]['elo']) ** 2)) ** 0.5))\
        if stats[player]['elo'] <= 2355 else 50.0
    # 
    effective_n = min(stats[player]['games'], n_star)
    #
    k = 800.0 / (effective_n + num_games)
    stats[player]['elo'] = stats[player]['elo'] + k * (stats[player]['p'] - stats[player]['e'])
    stats[player]['p'] = 0
    stats[player]['e'] = 0

def plot_elo_ratings(args, evaluation_output_file):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + evaluation_output_file + '.json'))
    with open(path, 'r') as json_file:  
        stats = json.load(json_file)
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + evaluation_output_file + '.html'))
    # 
    x = list(map(lambda x: args.save_point * (x - 1), range(1, len(stats))))
    y = list(map(lambda x: stats[x]['elo'], range(1, len(stats))))
    # print(x)
    # print(y)
    trace = graph_objs.Scatter(
        x = x,
        y = y,
        mode = 'lines+markers'
    )
    data = [trace]
    plotly.offline.plot(data, filename = path)

# ============================================================================================================================ #
# Point of entry
# ============================================================================================================================ #

if __name__ == '__main__':
    args = parse_args()
    run_model(args)
    if args.elo:
        evaluate_over_time(args, True)
