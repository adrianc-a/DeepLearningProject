# holds the game state in the first 9 positions then the current player
# at index 9 and the move count at index 10 
STARTING_CONFIG = ['.', '.', '.', '.', '.', '.', '.', '.', '.', 0, 0]

ICONS = ['X', 'O']

STARTING_MOVES = []

class TTT():
   
    def __init__(self, board_rep=STARTING_CONFIG, move_stack=STARTING_MOVES):
        self.board = board_rep[:]
        self.move_stack = move_stack[:] #really just to maintain parity with the way python chess works
    
    def __repr__(self):
        res = '' 
        for i in range(3):
            for j in range(3):
                res += self.board[3*i + j]
            res += '\n'
        return res
   
    def copy(self):
       return TTT(self.board, self.move_stack) 

    @property
    def legal_moves(self):
        return (i for i in range(9) if self.board[i] == '.')

    def push(self,move_idx):
        lmvs = len(self.move_stack) 
        player_idx =  self.board[9]
        
        if move_idx >= 0 and move_idx < 9 and self.board[move_idx] == '.':
            self.move_stack.append(move_idx)
            self.board[10] += 1
            self.board[move_idx] = ICONS[player_idx]
            self.board[9] = (player_idx + 1) % 2
        else:
            raise ValueError("Attempted to play invalid move")

    @property
    def current_player(self):
        return self.board[9]

    @property
    def move_count(self):
        return self.board[10]

    def draw(self):
        return all(board[i] != '.' for i in range(9))

    def is_three_in_a_row(self): 
        return tiar(self,0) or tiar(self,1)
        
    def tiar(self,player_idx):
        tiar = False

        icon = ICONS[player_idx]

        #check horizontal matches
        for i in range(3):
            tiar = tiar or all(self.board[i * 3 + j] == icon for j in range(3))
     
        #check vertical matches 
        for i in range(3):
            tiar = tiar or all(self.board[i + 3 * j] == icon for j in range(3))
       
        tiar = tiar or all(self.board[i] == icon for i in [0,4,8])
        tiar = tiar or all(self.board[i] == icon for i in [2,4,6])

        return tiar

    @property
    def turn(self):
        return self.board[9] 

    def num_full_moves(self):
        return self.board / 2

    def pop(self):
        if self.move_count == 0: raise ValueError("No more available moves to pop")

        self.board[10] -= 1
        self.player = (self.board[9] + 1) % 2
        self.board[self.move_stack.pop()] = '.'

