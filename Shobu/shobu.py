'''
A bitboard implementation of the board game 'Shōbu'.
'''
from random import choice

DIRECTIONS = {'N':5, 'E':-1, 'S':-5, 'W':1, 'NE':4, 'NW': 6, 'SE': -6, 'SW': -4}
BITMASK = 0b01111011110111101111 # bitmask deletes bits that move off the board

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
    print(*mailbox_2D, sep='\n', end=f'\n{message}\n\n')
    return

def dilate(x, direction, distance):
    for _ in range(distance):
        x |= bitshift(x,direction,1)
    return x

def erode(x, direction, distance):
    for _ in range(distance):
        x &= bitshift(x,direction,1)
    return x

class Board:
    def __init__(self, type: bool, bitboard1: int = 0b1111000000000000000, bitboard2: int = 0b1111) -> None:
        self.type = type # light-board = False, dark-board = True
        self.bits = [bitboard2, bitboard1] # index 1 corresponds to player 1

    def mailbox(self):
        'Transform the bitboards into a 2D mailbox representation.'
        if self.type:
            colour = '\033[34m' # blue
        else:
            colour = '\033[39m' # white
        mailbox_2D = 4 * [colour]
        for row in range(4):
            for n in range(5*row, 5*row+5):
                mailbox_2D[row] += str( (self.bits[1] & (1 << n) and 1) + 2 * (self.bits[0] & (1 << n) and 1) )
        return mailbox_2D


class State:
    'Object containing all information required to uniquely define a Shōbu game state.'
    def __init__(self, player: bool = True, side1: list[Board] = [Board(False), Board(True)], side2: list[Board] = [Board(True), Board(False)]) -> None:
        self.player = player # player 1 = True, player 2 = False
        self.boards = [side2, side1] # index 1 corresponds to player 1

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        print(f'{message}----------')
        for side in [ self.boards[self.player], self.boards[not self.player] ]:
            left = side[0].mailbox()
            right = side[1].mailbox()
            for row in range(4):
                print(left[row]+right[row])
            print('')
        return
    
    def is_game_over(self) -> int:
        '''The first player to lose all their stones on any board loses the game. 
        Return values 1 = player 1 win, -1 = player 2 win and 0 = game not over.'''
        if not self.boards[0][0].bits[0] or not self.boards[0][1].bits[0]:
            return 1 # player 1 wins
        elif not self.boards[1][0].bits[1] or not self.boards[1][1].bits[1]: 
            return -1 # player 2 wins
        return 0 # game not over
    
    def move(self, board1: Board, board2: Board, end_square: int, direction: int, distance: int):
        start_square = bitshift(end_square, -direction, distance)
        board1[self.player] ^= start_square|end_square
        board2[self.player] ^= start_square|end_square
        path = dilate(start_square,direction,distance)
        collision = path & board2[not self.player]
        if collision: 
            landing = bitshift(end_square, direction, 1)
            board2[not self.player] ^= collision|landing
        return
    
    def ask_for_move(self, homeboard: int, aggroboard: int, end_square: int, direction: int, distance: int):
        home = input('Which passive board 1-4?')
        aggro = input('Which aggro board 1-4?')
        answer = int
        match answer:
            case 1:
                board1 = self.boards[not self.player][0]
            case 2:
                board2 = self.boards[not self.player][1]
            case 3:
                board3 = self.boards[self.player][0]
            case 4:
                board4 = self.boards[self.player][1]
        return

    def all_legal_moves(self):
        moves = {}
        o1, p1 = self.boards[self.player]
        o2, p2 = self.boards[not self.player]
        for key, direction in DIRECTIONS.items():
            legals = self.legal_moves(direction, p1, p2, o1, o2)
            if legals:
                moves[key] = legals
        return moves

    def legal_moves(self, direction: int, board1: Board, board2: Board) -> dict[int,int]:
        legals = {}
        legal_passive = ~(board1[self.player] | board1[not self.player])
        passive1 = bitshift(board1[self.player], direction, 1) & legal_passive
        if passive1: # can play a passive move at distance 1
            legal_aggro = ~erode(board2[self.player] | board2[not self.player], -direction, 1) & ~bits_p2
            aggro1 = bitshift(board2[self.player], direction, 1) & legal_aggro
            legal1 = passive1 & aggro1 
            if legal1: # has legal moves at distance 1
                legals[1] = legal1
                passive2 = bitshift(passive1, direction, 1) & legal_passive
                if passive2: # can play a passive move at distance 2
                    aggro2 = bitshift(aggro1, direction, 1) & legal_aggro
                    legal2 = passive2 & aggro2
                    if legal2:
                        legals[2] = legal2
        return legals
    
    @staticmethod
    def random_move(all_legal_moves: dict[ str, dict[int,int] ]) -> int:
        direction = choice( list(all_legal_moves) )
        distance = choice( list(all_legal_moves[direction]) )
        random_moves = split( all_legal_moves[direction][distance] )
        return choice(random_moves)

if __name__ == '__main__':
    game = State()
    game.render()
    game.is_game_over()
    pass