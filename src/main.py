import players
import alphago as ag
import argparse
from game import Game
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager
from chess_manager import ChessManager
from networks import NetworkWrapper, OPTIMIZER_REG
from sys import argv
import tensorflow as tf
import numpy as np

from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train/play games with alphago techniques'
    )

    parser.add_argument('-t', '--train-model', action='store_true')
    parser.add_argument('-l', '--load-model', action='store_true')

    parser.add_argument('-g', '--game', choices=['ttt', 'c4', 'chess'], required=True)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-n', '--num_games', type=int)
    parser.add_argument('-a', '--learning_rate', type=float)
    parser.add_argument('-p', '--play-game', choices=['alphago', 'human', 'simple', 'random'], nargs=2)
    parser.add_argument('-s', '--save-model', action='store_true')
    parser.add_argument('-f', '--save-file', type=str)
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'adam'])

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
        NetworkWrapper.restore(path, shape, opt)
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


def run_mode(args):
    if args.train_model:
        ag_player = train_model(args.game, args.iterations, args.num_games)
    elif args.load_model:
        opt = OPTIMIZER_REG[args.optimizer](learning_rate=args.learning_rate)
        ag_player = load_model(args.game, args.save_file, opt)

    #show_graph(tf.get_default_graph().as_graph_def())
    print(ag_player.nn.forward(TicTacToeManager().state2vec()))
    #print(ag_player.nn.sess.graph.get_tensor_by_name('conv2d_1/kernel:0').eval(session=ag_player.nn.sess))
    #print(ag_player.nn.sess.graph.get_tensor_by_name('batch_normalization_1/keras_learning_phase').eval(session=ag_player.nn.sess))
    if args.play_game:
        p1 = get_players(args.game, args.play_game[0], ag_player)
        p2 = get_players(args.game, args.play_game[1], ag_player)

        Game(get_manager(args.game), p1, p2).play()

    if args.save_model:
        ag_player.nn.save(args.save_file)


if __name__ == '__main__':
    run_mode(parse_args())
