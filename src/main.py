import players
from game import Game
from tictactoe_manager import TicTacToeManager
from connect4 import Connect4Manager
from chess_manager import ChessManager

# g = Game(ChessManager(), players.chess_human_player, players.random_player)
g = Game(ChessManager(), players.mcts_player, players.mcts_player)
g.play()




# This is how I think the flow should be
# ag = AlphaGoZeroArchitectures.chess()
# trainer = AlphaGoTrainer(ag)
# trainer.train(...)
# g = Game(ChessManager(), players.chess_human_player, ag.play_move)
# g.play()
