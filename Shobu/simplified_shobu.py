DIRECTIONS = {'N':5, 'E':-1, 'S':-5, 'W':1, 'NE':4, 'NW': 6, 'SE': -6, 'SW': -4}
BITMASK = 0b01111011110111101111 # bitmask deletes bits that move off the board

class Board:
    '''Bitboard representation of one of the game boards.'''
    def __init__(self) -> None:
        self.bitboards = [0b1111, 0b1111000000000000000] # index 1 corresponds to the bitboard of player 1

    def mailbox(self) -> list[list[int]]:
        'Transform two bitboards into a 2D mailbox representation and print it.'
        mailbox = [ (self.bitboards[1] & (1 << n) and 1) + 2*(self.bitboards[0] & (1 << n) and 1) for n in range(20) ]
        mailbox_2D = [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]
        print(*mailbox_2D, sep='\n', end='\n\n')
        return mailbox_2D
    
    def empty(self) -> int:
        return ~(self.bitboards[0] | self.bitboards[1])

class Shobu:
    def __init__(self) -> None:
        self.player = True # player 1 = True, player 2 = False
        self.boards = [Board(), Board()]

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        [b.mailbox() for b in self.boards]
        print(f'{message}----------------')
        return
    
    def is_game_over(self):
        'Check if all of a players\' stones have been removed from any of the boards. If so, that player loses.'
        for board in self.boards:
            if not board.bitboards[1]: 
                return -1 # player 2 wins
            elif not board.bitboards[0]: 
                return 1 # player 1 wins
        return 0 # game not over
    
    def passive_move(self, homeboard: Board, end_square: int, direction: int, distance: int) -> None:
        'Updates a board with a passive move, assuming it is legal.'
        start_square =  self.bitshift(end_square, -direction, distance)
        homeboard.bitboards[self.player] ^= start_square|end_square
        return
    
    def legal_passives(self, homeboard: Board) -> dict[str,dict[int,int]]:
        '''Calculate the ending squares of legal passive moves in all directions. 
        To access `moves` from returned dictionary use dictionary[`direction`][`distance`].'''
        all_legal_moves = {}
        for key, direction in DIRECTIONS.items():
            moves = homeboard.bitboards[self.player]
            legal_moves = {}
            for distance in [1,2]:
                moves = homeboard.empty() & self.bitshift(moves, direction, 1)
                if moves:
                    legal_moves[distance] = moves
            if legal_moves:
                all_legal_moves[key] = legal_moves      
        return all_legal_moves
    
    @staticmethod
    def bitshift(bits: int, direction: int, distance: int) -> int:
        shift = direction * distance
        if shift > 0:
            return (bits >> shift) & BITMASK
        else:
            return (bits << -shift) & BITMASK
    
    @staticmethod
    def split(legal_moves: int) -> list[int]:
        'Split `legal moves` into invidual moves by the on-bits.'
        seperate_moves = []
        while legal_moves:
            seperate_moves.append(legal_moves & -legal_moves) # get least significant on-bit
            legal_moves &= legal_moves-1 # remove least significant on-bit
        return seperate_moves

def view(bits: int, message: str = '') -> None:
    'For debugging.'
    mailbox = [ bits & (1 << n) and 1  for n in range(20) ]
    mailbox_2D = [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]
    print(*mailbox_2D, sep='\n', end=f'\n{message}\n')
    return
