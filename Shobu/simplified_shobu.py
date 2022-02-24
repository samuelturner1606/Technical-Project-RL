class Shobu:
    def __init__(self) -> None:
        self.board1 = Board()
        self.board2 = Board()

    def render(self) -> None:
        '''Print the current game state.'''
        print(*self.board1.mailbox(), sep='\n')
        print('\n')
        print(*self.board2.mailbox(), sep='\n')
        return

class Board:
    def __init__(self) -> None:
        self.bitboard1 = 0b01111000000000000000
        self.bitboard2 = 0b1111

    def mailbox(self) -> list[list[int]]:
        '''Transform two bitboards into a 2D mailbox representation.'''
        mailbox = [ (self.bitboard1 & (1 << n) and 1) + 2*(self.bitboard2 & (1 << n) and 1) for n in range(4*5) ]
        return [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]

s = Shobu()
s.render()