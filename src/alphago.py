from game import Game, GameResult
import networks as net

class AlphaGoZero:

    # nn must be game specific and line up according to a respective
    # state_manager
    def __init__(self, nn):
        self.nn = nn

    def play_move(current_state, next_states):
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
        pass

# there is a case to be made that the state2vec should be moved out of the
# state_manager and just put into a separate class/module which converts
# particular game states into vectors, which is linked to alphagozero since it
# is the only player that requires these vectorizations
# maybe this should be a module
class AlphaGoZeroArchitectures:

    # return a nn for alpha go zero based on the ttt game
    # that is, an instance of AlphaGoZero
    @staticmethod
    def ttt():
        """
        return AlphaGoZero(
            net.alphago_net(
                ...
            )
        )
        """
        pass

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
            self._end_game
        )

        for i in range(iterations):
            g.play(games)
            self.update_weights(sample_pct)

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

        batch_size = np.floor(pct * len(self.S))
        ind = np.random.choice(len(self.S) - 1, batch_size, replace=False)

        self.player.nn.training_step(
            np.array(self.S)[ind],
            np.array(self.P)[ind],
            np.array(self.Z)[ind]
        )


    def play_move(current_state, next_states):
        # since this is always called, regardless of player, we can keep the
        # states (s, pi, z)
        move_index = self.player.play_move(current_state, next_state)

        self.cur_S.append(current_state.state2vec())
        self.cur_P.append(self.player.p)

        return move_index

