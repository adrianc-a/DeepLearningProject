class Game:

    def __init__(self, manager, player1, player2):
        self.manager = manager
        self.player1 = player1
        self.player2 = player2

    def play(self, num_games=1):
        for i in range(num_games):
            self._play_game()

    def _play_game(self):
        self.manager.new_game()
        self.manager.output()

        # true means player1's turn, False player2
        turn = True

        while not self.manager.is_terminal_state():
            moves = self.manager.next_states()

            if len(moves) == 0:
                print('draw')
                return

            player = self.player1 if turn else self.player2

            move_idx = player(self.manager.current_state(), moves)

            if move_idx < 0 or move_idx >= len(moves):
                print('illegal move made')
                return

            self.manager = self.manager.make_move(move_idx)
            self.manager.output()
            turn = not turn
        if self.manager.is_win():
            print(
                'The winner is: ' +
                    ('player1' if self.manager.zero_is_winner() else 'player2')
            )
        elif self.manager.is_draw():
            print('Draw')
        else:
            print('Game ended')



