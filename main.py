import players
import game

g = game.TTT(players.ttt_simple_player, players.ttt_random_player)
g.play()
