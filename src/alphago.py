from game import Game, GameResult
import networks
import policy
from math import floor
from numpy import argmax
from numpy import random
from numpy import array
from mcts import MCTS

class AlphaGoZero:

    # nn must be game specific and line up according to a respective
    # state_manager
    def __init__(self, nn):
        self.nn = nn
        self.mcts = MCTS(policy.upper_confidence_bound, nn)

    def play_move(self, current_state, next_states):
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
        pi = self.mcts(current_state, floor(current_state.num_full_moves() / 2))

        ind = argmax(pi)
        self.pi = pi[ind]

        return ind


# there is a case to be made that the state2vec should be moved out of the
# state_manager and just put into a separate class/module which converts
# particular game states into vectors, which is linked to alphagozero since it
# is the only player that requires these vectorizations
# maybe this should be a module
class AlphaGoZeroArchitectures:


    @staticmethod
    def create_player(nn, opt):
        return AlphaGoZero(networks.NetworkWrapper(nn, opt))

    # can we have this use state2vec.shape somehow?
    @staticmethod
    def ttt_input_shape():
        return (3,3,3)

    # return a nn for alpha go zero based on the ttt game
    # that is, an instance of AlphaGoZero
    @staticmethod
    def ttt():
        return AlphaGoZeroArchitectures.create_player(
            networks.alphago_net(AlphaGoZeroArchitectures.ttt_input_shape(), 2, (2,2), 2, (1,1)),
            networks.OPTIMIZER_REG['sgd'](learning_rate=0.01)
        )

    # if we want to move the 2vec stuff to this class, so it lines up nicely
    # with the architectures
    @staticmethod
    def ttt_state2vec(state):
        pass


class AlphaGoZeroTrainer:
    def __init__(self, alphagozero_player):
        self.player = alphagozero_player
        self.S = []
        self.P = []
        self.Z = []

    def train(self, manager, iterations=10, games=10, sample_pct=0.75):
        g = Game(
            manager,
            self.play_move,
            self.play_move,
            self._begin_game,
            self._end_game,
            False,
            False
        )

        for i in range(iterations):
            g.play(games)
            self.update_weights(sample_pct)
            print('Finished iteration ' + str(i))

    def _begin_game(self):
        self.cur_S = []
        self.cur_P = []

    def _end_game(self, end_type, winner):
        # I'm not sure what to do if it's a draw/something else
        if end_type != GameResult.WIN:
            return

        winner = -1 if winner == 0 else 1

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

        self.player.nn.training_step(
            array(self.S)[ind],
            array(self.P)[ind],
            array(self.Z)[ind]
        )


    def play_move(self, current_state, next_states):
        # since this is always called, regardless of player, we can keep the
        # states (s, pi, z)
        move_index = self.player.play_move(current_state, next_states)

        state_vec = current_state.state2vec()
        self.cur_S.append(state_vec.reshape(state_vec.shape[1:]))

        # I assume that the play_move from AlphaGoZero, keeps this, I wish there
        # was a more clever way of doing this. NOTE: P is the probability which
        # corresponds to the move which was made, (referring to pi = [...]
        # which is returned by MCTS.__call__
        self.cur_P.append(self.player.pi)

        return move_index

