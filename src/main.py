import players
from game import Game
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager
from chess_manager import ChessManager
import alphago as ag
from os import getcwd
from networks import NetworkWrapper 



# save
ag_player = ag.AlphaGoZeroArchitectures.ttt()
ag_trainer = ag.AlphaGoZeroTrainer(ag_player)
ag_trainer.train(TicTacToeManager(), iterations=20, games=50)

ag_player.nn.save('./test')


"""
# recover
nn = NetworkWrapper.restore('./test/')

ag_player = ag.AlphaGoZero(nn)
"""




g = Game(TicTacToeManager(), players.ttt_human_player, ag_player.play_move)
g.play()
