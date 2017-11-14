# create empty, untrained ttt player
def ttt_alphago_zero_player():
    pass


def ttt_human_player(state):
    print('this is the board')
    print(state)
    move = input('what is your move? (i.e. x y)')
    return int(move[0]), int(move[2])


def ttt_simple_player(state):
    for i in range(state):
        for j in range(state[i]):
            if state[i][j] is None:
                return i, j
    return 0, 0
