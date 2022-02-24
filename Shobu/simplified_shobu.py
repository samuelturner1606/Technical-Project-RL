def _view(bitboard: int, message: str):
        '''Transform a bitboard into 2D mailbox representation for debugging purposes.'''
        mailbox = [ (bitboard & (1 << n) and 1) for n in range(4*5) ]
        print(f'\n{message}')
        return [ print(mailbox[ 5*row : 5*row+5 ]) for row in range(4) ]

class Board:
    '''Bitboard representation of one of the game boards.'''
    def __init__(self) -> None:
        self.bitboard1 = 0b01111000000000000000
        self.bitboard2 = 0b1111

    def mailbox(self) -> list[list[int]]:
        '''Transform two bitboards into a 2D mailbox representation.'''
        mailbox = [ (self.bitboard1 & (1 << n) and 1) + 2*(self.bitboard2 & (1 << n) and 1) for n in range(4*5) ]
        return [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]

class Shobu:
    def __init__(self) -> None:
        self.board1 = Board()
        self.board2 = Board()
        self.done = False
        self.player = True # player 1 = True, player 2 = False

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
        return # game not done
    
    def passive_move(self, homeboard: Board, start_square: int, direction: int, distance: int) -> None:
        '''Updates a board with a passive move, assuming it is legal.'''
        end_square =  int(start_square * 2**(direction*distance))
        if self.player == 1:
            homeboard.bitboard1 ^= start_square|end_square
        else:
            homeboard.bitboard2 ^= start_square|end_square
        return

s = Shobu()
DIRECTIONS = {'N':-5, 'E':1, 'S':5, 'W':-1, 'NE':-4, 'NW': -6, 'SE': 6, 'SW': 4}
BITMASK = 0b01111011110111101111
s.passive_move(s.board2, 0b0100000000000000000, DIRECTIONS['NW'], 2)