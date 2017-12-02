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

from IPython.display import clear_output, Image, display, HTML

def parse_args():
	parser = argparse.ArgumentParser(description='Train/play games with alphago techniques')
	parser.add_argument('-t', '--train-model', action='store_true')
	parser.add_argument('-l', '--load-model', action='store_true')
	parser.add_argument('-g', '--game', choices=['ttt', 'c4', 'chess'], required=True)
	parser.add_argument('-i', '--iterations', type=int)
	parser.add_argument('-n', '--num_games', type=int)
	parser.add_argument('-a', '--learning_rate', type=float)
	parser.add_argument('-p', '--play-game', action='store_true')
	parser.add_argument('-q', '--players', choices=['alphago', 'human','simple', 'random'], nargs=2)
	parser.add_argument('-s', '--save-model', action='store_true')
	parser.add_argument('-f', '--save-file', type=str)
	parser.add_argument('-o', '--optimizer', choices=list(networks.OPTIMIZER_REG.keys()))
	parser.add_argument('-e', '--eval', action='store_true')
	parser.add_argument('-m', '--num_mctsrun', type=int)
	parser.add_argument('-tc', '--temp_change_iter', type=int)
	parser.add_argument('-te', '--temp_early', type=int)
	parser.add_argument('-tl', '--temp_late', type=float)
	parser.add_argument('-b', '--num_train_step_batch', type=int)
	return parser.parse_args(argv[1:])

def train_model(game, iterations, num_games):
    if game == 'ttt':
        player = ag.AlphaGoZeroArchitectures.ttt()
    elif game == 'c4':
        player = ag.AlphaGoZeroArchitectures.c4()
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
        shape = ag.AlphaGoZeroArchitectures.c4_input_shape()
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

		
		
def run_mode(argss):
    global args
    args=argss
    if args.train_model:
        ag_player = train_model(args.game, args.iterations, args.num_games)
    elif args.load_model:
        opt = OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate)
        ag_player = load_model(args.game, args.save_file, opt)
    else:
        ag_player = None

    if args.play_game or args.eval:
        p1 = get_players(args.game, args.players[0], ag_player)
        p2 = get_players(args.game, args.players[1], ag_player)


    if args.play_game:
        Game(get_manager(args.game), p1, p2).play()

    if args.eval:
        print(Evaluator(get_manager(args.game), p1, p2).evaluate())


    if args.save_model:
        ag_player.nn.save(args.save_file)


if __name__ == '__main__':
    run_mode(parse_args())
