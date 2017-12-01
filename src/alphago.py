from game import Game, GameResult, Evaluator
import networks
from math import floor
from numpy import argmax
from numpy import random
from numpy import array
from mcts import MCTS
import chess_manager
import tictactoe_manager
import connect4

class AlphaGoZero:

    # nn must be game specific and line up according to a respective
    # state_manager
    def __init__(self, nn, man):
        self.nn = nn
        self.mcts = MCTS(nn, man)

    def notify_move(self, move_idx):
        self.mcts.set_root(move_idx)

    def play_move(self, current_state, next_states, is_train=False):
        """
        here we can explore all next states
        vectorize each state
        for each next_state we can explore it's next_states too
        this should be sufficient for mcts

        we can also support different types of architectures. i.e. the original
        one where the output is moves, or the one where we pass each next_state
        into a nn and get some rating, simply by specificying the type either in
        the nn module/class or here, as a parameter in __init__
        """

        # number of moves should not be a param, mcts should infer it
        # since not implemented i expect a normal python array of floats

        pi = self.mcts(current_state, n = 20)

        if is_train:
            ind = random.choice(len(pi), p=pi)
        else:
            ind = argmax(pi)

        self.mcts.set_root(ind)

        self.pi = pi
        return ind

# there is a case to be made that the state2vec should be moved out of the
# state_manager and just put into a separate class/module which converts
# particular game states into vectors, which is linked to alphagozero since it
# is the only player that requires these vectorizations
# maybe this should be a module
class AlphaGoZeroArchitectures:

    @staticmethod
    def create_player(nn, opt, man):
        return AlphaGoZero(networks.NetworkWrapper(nn, opt), man)


    # can we have this use state2vec.shape somehow?
    @staticmethod
    def ttt_input_shape():
        return tictactoe_manager.INPUT_SHAPE

    # return a nn for alpha go zero based on the ttt game
    # that is, an instance of AlphaGoZero
    @staticmethod
    def ttt():
        return AlphaGoZeroArchitectures.create_player(
            networks.alphago_net(AlphaGoZeroArchitectures.ttt_input_shape(), 4, (1,1), 5, (2,2)),
            networks.OPTIMIZER_REG['sgd'](learning_rate=0.01),
            tictactoe_manager.TicTacToeManager()
        )

    @staticmethod
    def chess_input_shape():
        return chess_manager.INPUT_SHAPE

    @staticmethod
    def chess_net():
        return AlphaGoZeroArchitectures.create_player(
            #the residual and conv blocks have 256 layers and there are 10 conv blocks
            networks.alphago_net(AlphaGoZeroArchitectures.chess_input_shape(), 256, (3,3), 10, (3,3)),
            networks.OPTIMIZER_REG['sgd'](learning_rate=0.01)
        )


    @staticmethod
    def connect4_net():
        return AlphaGoZeroArchitectures.create_player(
            #networks.alphago_net(AlphaGoZeroArchitectures.chess_input_shape(), 128, (3,3), 10, (3,3)),
            #networks.OPTIMIZER_REG['sgd'](learning_rate=0.01)
            networks.alphago_net(AlphaGoZeroArchitectures.connect4_input_shape(), 256, (3,3), 10, (3,3)),
            networks.OPTIMIZER_REG['sgd'](learning_rate=0.01)
        )

    @staticmethod
    def connect4_input_shape():
        return connect4.INPUT_SHAPE

    @staticmethod
    def from_checkpoint(path, shape, opt, man):
        return AlphaGoZero(networks.NetworkWrapper.restore(path, shape, opt), man)

    @staticmethod
    def shape(game):
        if game == 'ttt':
            return AlphaGoZeroArchitectures.ttt_input_shape()
        elif game == 'c4':
            return AlphaGoZeroArchitectures.connect4_input_shape()
        else:
            return AlphaGoZeroArchitectures.chess_input_shape()

    @staticmethod
    def get_manager(game):
        if game == 'ttt':
            return tictactoe_manager.TicTacToeManager()
        elif game == 'c4':
            return connect4.Connect4Manager()
        else:
            return chess_manager.ChessManager()


class AlphaGoZeroTrainer:
    def __init__(self, alphagozero_player, name=''):
        self.player = alphagozero_player
        self.S = []
        self.P = []
        self.Z = []
        self.path = 'models/'
        self.name = name

    def train(self, manager, iterations=10, games=10, sample_pct=0.85, ckpt=15):
        self.path += manager.name() + '_' + self.name
        self.game = manager.name()

        for i in range(1, iterations + 1):
            g = Game(
                manager,
                player1=self.play_move,
                player2=self.play_move,
                begin_play=self._begin_play,
                begin_game=self.player.mcts._begin_game,
                end_game=self._end_game,
                log=False,
                render=False
            )
            g.play(games)
            self.update_weights(sample_pct)
            print('Finished iteration ' + str(i))

            if i % ckpt == 0:
                self.player.nn.save(self.path + '_' + str(i))
                # check it's not the 1st checkpoint
                if i - ckpt > 0:
                    self.player = self._evaluate_cur_player(i - ckpt, i)
                    #self.player.mcts._begin_game()

    def _eval_begin_game(self, prev, cur):
        print('ok')
        prev.mcts._begin_game()
        cur.mcts._begin_game()

    def _evaluate_cur_player(self, prev_i, cur_i):
        prev_player = AlphaGoZeroArchitectures.from_checkpoint(
            self.path + '_' + str(prev_i),
            AlphaGoZeroArchitectures.shape(self.game),
            #networks.OPTIMIZER_REG['sgd'](learning_rate=0.01)
            self.player.nn.optimizer,
            AlphaGoZeroArchitectures.get_manager(self.game)
        )

        winner, _ = Evaluator(
            AlphaGoZeroArchitectures.get_manager(self.game),
            prev_player.play_move,
            self.player.play_move,
            begin_game=lambda: self._eval_begin_game(prev_player, self.player)
        ).evaluate()

        if winner == 0:
            print('checkpoint using previous')
            self.player.nn.sess.close()
            return prev_player
        else:
            print('checkpoint using current')
            prev_player.nn.sess.close()
            return self.player



    def _begin_play(self):
        self.cur_S = []
        self.cur_P = []

    def _end_game(self, end_type, winner):
        # I'm not sure what to do if it's a draw/something else
        if end_type != GameResult.WIN:
            return

        winner = 1 if winner == 0 else -1

        flip = lambda x: -1 if x == 1 else 1

        # there is probably a more elegant method for adding these
        # cur_Z is the winner, relative to the current player
        cur_Z = []
        for i in range(len(self.cur_S)):
            cur_Z.append(winner)
            winner = flip(winner)
        cur_Z.reverse()

        # add the triple's for the latest game to the total data
        self.S.extend(self.cur_S)
        self.P.extend(self.cur_P)
        self.Z.extend(cur_Z)


    def update_weights(self, pct):
        # based on the games and self.data
        # update the weights of the nn

        batch_size = floor(pct * len(self.S))
        ind = random.choice(len(self.S) - 1, batch_size, replace=False)


        self.S = array(self.S)[ind].tolist()
        self.P = array(self.P)[ind].tolist()
        self.Z = array(self.Z)[ind].tolist()

        batch_S = []
        batch_P = []
        batch_Z = []

        for i in range(len(self.S)):
            for j in range(len(self.S[i])):
                batch_S.append(self.S[i][j])
                batch_P.append(self.P[i][j])
                batch_Z.append(self.Z[i])



        self.player.nn.training_step(
            array(batch_S),
            array(batch_P),
            array(batch_Z)
        )


    def play_move(self, current_state, next_states):
        # since this is always called, regardless of player, we can keep the
        # states (s, pi, z)
        move_index = self.player.play_move(
            current_state, next_states,
            is_train=True
        )

        state_vec, managers = current_state.moves2vec()
        self.cur_S.append(state_vec)

        # I assume that the play_move from AlphaGoZero, keeps this, I wish there
        # was a more clever way of doing this. NOTE: P is the probability which
        # corresponds to the move which was made, (referring to pi = [...]
        # which is returned by MCTS.__call__
        self.cur_P.append(self.player.pi)

        return move_index
