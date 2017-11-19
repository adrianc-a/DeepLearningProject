import players
from game import Game
from tictactoe_manager import TicTacToeManager

g = Game(TicTacToeManager(), players.ttt_simple_player, players.ttt_random_player)
g.play()

