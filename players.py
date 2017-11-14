from random import randint

# create empty, untrained ttt player
def ttt_alphago_zero_player():
    pass


def ttt_human_player(state):
    move = input('what is your move? (i.e. x y)\n')
    return int(move[0]), int(move[2])

def ttt_random_player(state):
    moves = [
        (i, j) for i in range(len(state)) for j in range(len(state))
        if state[i][j] == ' '
    ]
    return moves[randint(0, len(moves) - 1)]

def ttt_simple_player(state):
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] is ' ':
                return i, j
    return 0, 0
