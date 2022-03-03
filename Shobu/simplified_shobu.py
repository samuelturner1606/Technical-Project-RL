DIRECTIONS = {'N':5, 'E':-1, 'S':-5, 'W':1, 'NE':4, 'NW': 6, 'SE': -6, 'SW': -4}
BITMASK = 0b01111011110111101111 # bitmask deletes bits that move off the board

class Board:
    '''Bitboard representation of one of the game boards.'''
    def __init__(self, bitboard1: int = 0b1111000000000000000, bitboard2: int = 0b1111) -> None:
        self.bitboards = [bitboard2, bitboard1] # index 1 corresponds to the bitboard of player 1

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
        start_square = bitshift(end_square, -direction, distance)
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
                moves = homeboard.empty() & bitshift(moves, direction, 1)
                if moves:
                    legal_moves[distance] = moves
            if legal_moves:
                all_legal_moves[key] = legal_moves      
        return all_legal_moves
 
def bitshift(bits: int, direction: int, distance: int) -> int:
    shift = direction * distance
    if shift > 0:
        return (bits >> shift) & BITMASK
    else:
        return (bits << -shift) & BITMASK
    

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

def dilate(x, direction, dist):
    for _ in range(dist):
        x |= bitshift(x,direction,1)
    return x

def erode(x, direction, dist):
    for _ in range(dist):
        x &= bitshift(x,direction,1)
    return x

def legal_aggros1(board: Board, player: bool, direction: int):
    bits_p = board.bitboards[player]
    bits_o = board.bitboards[not player]
    return ~bits_p & bitshift(bits_p, direction, 1) & ~erode(bits_o | bits_p,-direction,1)

# parity
s = Shobu()
b1 = 0b1001111
b2 = 0b110011010110000000
dis = 2
dr = DIRECTIONS['S']

'''
a = bitshift(b1,dr,1) & b2
b = bitshift(b1,dr,2) & b2
c = bitshift(b1,dr,3) & b2

d = bitshift(b1,dr,dis-1) ^ a
e = bitshift(d,dr,dis-1) ^ b
f = bitshift(e,dr,dis-1) ^ c
'''
#
import random
b1 = random.getrandbits(20) & BITMASK
b2 = random.getrandbits(20) & BITMASK
b2 ^= b1 & b2
s.boards[1].bitboards = [b2, b1]
s.boards[0].bitboards = [b2, b1]
s.render()
a = legal_aggros1(s.boards[0],True,dr) 

change = dilate(a,-dr,1)
change2 = dilate(a & b2,dr,1)
new1 = change ^ b1
new2 = change2 ^ b2
s.boards[0].bitboards = [new2, new1]
b = legal_aggros1(s.boards[0],True,dr) 
legal2 = bitshift(a & bitshift(b,-dr,1),dr,1)
s.boards[1].bitboards = [0, legal2]
s.render()

