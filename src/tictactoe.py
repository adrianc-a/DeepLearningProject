
STARTING_CONFIG = ['.', '.', '.', '.', '.', '.', '.', '.', '.']

class TTT():


    
    def __init__(self, board_rep=STARTING_CONFIG):
        self.board = board_rep[:]
        self.move_stack = []

    
    def __repr__(self):
        res = '' 
        for i in range(3):
            for j in range(3):
                res += self.board[3*i + j]
            res += '\n'

        return res

    @property
    def legal_moves(self):
        return (idx for idx,_ in enumerate(self.board))


    def push_move(self,move_idx):
        
        if 
