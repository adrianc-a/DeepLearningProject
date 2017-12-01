from random import randint

import chess

def simple_player(state, moves, previous_move):
    return 0

def random_player(state, moves, previous_move):
    return randint(0, len(moves) - 1)

def ttt_human_player(state, moves, previous_move):
    inp = input('Your move: r c:\n')
    row = int(inp[0])
    col = int(inp[2])
    move = 3 * row + col

    for m in range(len(moves)):
        board = moves[m].board.board
        if board[move] != '.':
            return m
    return 0

def connect_human_player(state, moves, previous_move):
    inp = input('Your move: col:\n')
    col_i = int(inp[0])

    board = state.state.state

    col = lambda b, c: [b[i][c] for i in range(6)]
    height = lambda c: len(list(filter(lambda i: i == ' ', c)))

    cur_col = col(board, col_i)
    cur_height = height(cur_col)

    for m in range(len(moves)):
        board = moves[m].state.state
        move_col = col(board, col_i)
        move_height = height(move_col)

        if cur_height != move_height:
            return m
    return 0

def chess_human_player(state, moves, previous_move):
    inp = input('Your move as uci: \n')

    move = chess.Move.from_uci(inp)

    return state.get_moves().index(move)
