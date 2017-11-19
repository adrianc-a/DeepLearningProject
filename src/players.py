from random import randint


def ttt_simple_player(state, moves):
    return 0

def ttt_random_player(state, moves):
    return randint(0, len(moves) - 1)

def ttt_human_player(state, moves):
    inp = input('Your move: r c:\n')
    row = int(inp[0])
    col = int(inp[2])
    move = 3 * row + col

    for m in range(len(moves)):
        board = moves[m].board.board
        if board[move] != '.':
            return m
    return 0
