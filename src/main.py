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
from glob import glob

GAMES = ['ttt', 'c4', 'chess']
PLAYERS = ['alphago', 'human', 'simple', 'random']


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
    parser.add_argument('-q', '--players', choices=PLAYERS, nargs=2)
    parser.add_argument('-s', '--save-model', action='store_true')
    parser.add_argument('-f', '--save-file', type=str)
    parser.add_argument('-o', '--optimizer', choices=list(OPTIMIZER_REG.keys()))
    parser.add_argument('-e', '--eval', action='store_true')
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
        return ag.AlphaGoZeroArchitectures.chess(opt, args)


def run_mode(args):
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
        print(Evaluator(get_manager(args), p1, p2).evaluate())

    if args.save_model:
        ag_player.nn.save(args.save_file)


if __name__ == '__main__':
    run_mode(parse_args())
