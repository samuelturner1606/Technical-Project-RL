class Shobu:
    def __init__(self) -> None:
        self.board1 = Board()
        self.board2 = Board()
        self.done = False

    def render(self) -> None:
        '''Print the current game state.'''
        print(*self.board1.mailbox(), sep='\n')
        print('\n')
        print(*self.board2.mailbox(), sep='\n')
        return
    
    def is_game_over(self):
        '''Check if all of a players' stones have been removed from any of the boards.'''
        for board in [self.board1, self.board2]:
            if not board.bitboard1: # player 2 wins
                self.done = True
                return -1
            elif not board.bitboard2: # player 1 wins
                self.done = True
                return 1
        return # game still 

class Board:
    def __init__(self) -> None:
        self.bitboard1 = 0b01111000000000000000
        self.bitboard2 = 0b1111

    def mailbox(self) -> list[list[int]]:
        '''Transform two bitboards into a 2D mailbox representation.'''
        mailbox = [ (self.bitboard1 & (1 << n) and 1) + 2*(self.bitboard2 & (1 << n) and 1) for n in range(4*5) ]
        return [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]

s = Shobu()
s.board2.bitboard1 = 0
s.render()
print(s.is_game_over())