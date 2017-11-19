import players
from game import Game
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager

g = Game(Connect4Manager(), players.connect_human_player, players.random_player)
g.play()

