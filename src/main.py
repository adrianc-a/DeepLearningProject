import players
from game import Game
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager
from chess_manager import ChessManager
import alphago as ag


ag_player = ag.AlphaGoZeroArchitectures.ttt()
ag_trainer = ag.AlphaGoZeroTrainer(ag_player)
ag_trainer.train(TicTacToeManager())

g = Game(TicTacToeManager(), players.ttt_human_player, ag_player.play_move)
g.play()
