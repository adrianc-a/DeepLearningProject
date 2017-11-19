from random import randint


def ttt_simple_player(state, moves):
    return 0

def ttt_random_player(state, moves):
    return randint(0, len(moves) - 1)


