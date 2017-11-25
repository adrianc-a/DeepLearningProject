import policy
import mcts

class AlphaGoZero:

    # nn must be game specific and line up according to a respective
    # state_manager
    def __init__(self, nn):
        self.nn = nn
		self.simulator = mcts.MCTS(tree_policy = policy.upper_confidence_bound)
   
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
        move_vector=state.get_moves().index(simulator(state_manager = state))

# there is a case to be made that the state2vec should be moved out of the
# state_manager and just put into a separate class/module which converts
# particular game states into vectors, which is linked to alphagozero since it
# is the only player that requires these vectorizations
# maybe this should be a module
class AlphaGoZeroArchitectures:

    # return a nn for alpha go zero based on the ttt game
    @staticmethod
    def ttt_nn():
        pass

    # if we want to move the 2vec stuff to this class, so it lines up nicely
    # with the architectures
    @staticmethod
    def ttt_state2vec(state):
        pass


class AlphaGoZeroTrainer:
    def __init__(self, manager, nn):
        self.manager = manager
        self.player = AlphaGoZero(nn)
        self.triples = []

    def train(self, iterations=10, games=10):
        g = Game(manager, self.play_move, self.play_move)

        for i in range(iterations):
            g.play(games)
            self.update_weights()

    def update_weights(self):
        # based on the games and self.triples
        # update the weights of the nn
        pass

    def play_move(current_state, next_states):
        # since this is always called, regardless of player, we can keep the
        # states (s, pi, z)
        self.triples.append((current_state.state2vec(), None, None))
        self.player.play_move(current_state, next_state)

