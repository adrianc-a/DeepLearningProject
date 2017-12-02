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

# import plotly.plotly as plotly
# import plotly.graph_objs as graph_objs

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
    # 
    player = get_players(args.game, args.players[0], ag_player)
    # 
    model_directory = os.path.asbpath(os.path.join(os.path.dirname(__file__), '../model'))
    # get all checkpoints for this game and model
    checkpoints = []
    for directory in os.listdir(model_directory):
        if os.path.isdir(os.path.join(model_directory, directory)):
            if directory.startsWith(args.game + '_' + args.model_name + '_'):
                checkpoints.add(os.path.join(model_directory, directory))
    # 
    evaluation_file = args.game + '_' + args.model_name + '_' + args.players[0]
    # sort lexicographically
    checkpoints = sorted(checkpoints)
    for checkpoint in checkpoints:
        checkpoint_number = checkpoint.split('_')[-1]
        opt = OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate)
        ag_player = load_model(args.game, checkpoint, opt)
        # 
        Evaluator(get_manager(args.game), ag_player, player,
            player1_notify = player1_notify, player2_notify = player2_notify,
            player1_name = args.game + '_' + args.model_name + '_' + checkpoint_number, player2_name = args.players[0],
            should_rate = True, rate_after_each_game = False, evaluation_output = evaluation_file).evaluate()
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/' + evaluation_output + '.json'))
    rating = {}
    with open(path, 'r') as json_file:
        ratings = json.load(json_file)
    for checkpoint in ratings:
        checkpoint_number = checkpoint.split('_')[-1]
        print(checkpoint, ratings[args.game + '_' + args.model_name + '_' + checkpoint_number]['elo'])
    #TODO: plot this later

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
