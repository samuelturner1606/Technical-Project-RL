'''
A simplified version of the board game 'Shōbu' played on only two of the usual four boards. 
The game logic is implemented using bitboards.
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

class State:
    'Object containing all information required to uniquely define a simplified Shōbu board game state.'
    def __init__(self, player: bool = True, board1: list[int] = [0b1111, 0b1111000000000000000], board2: list[int] = [0b1111, 0b1111000000000000000]) -> None:
        self.player = player # player 1 = True, player 2 = False
        self.boards = [board2, board1] # index 1 corresponds to player 1

    def render(self, message1: str = '', message2: str = '') -> None:
        'Transform the two boards into 2D mailbox representation and print them.'
        print(f'{message1}----------------{message2}')
        for board in self.boards:
            mailbox = [ (board[1] & (1 << n) and 1) + 2 * (board[0] & (1 << n) and 1) for n in range(20) ]
            mailbox_2D = [ mailbox[ 5 * row : 5 * row + 5 ] for row in range(4) ]
            print(*mailbox_2D, sep='\n', end='\n\n')
        return
    
    def is_game_over(self):
        'Check if all of a players\' stones have been removed from any of the boards. If so, that player loses.'
        for board in self.boards:
            if not board[0]: 
                return 1 # player 1 wins
            elif not board[1]: 
                return -1 # player 2 wins
        return 0 # game not over
    
    def move(self, end_square: int, direction: int, distance: int) -> None:
        'Assuming it is legal, update both boards with a move.'
        start_square = bitshift(end_square, -direction, distance)
        # passive move
        self.boards[self.player][self.player] ^= start_square|end_square
        # aggro move
        aggro_board = self.boards[not self.player]
        aggro_board[self.player] ^= start_square|end_square

        path = dilate(start_square,direction,distance)
        collision = path & aggro_board[not self.player]
        if collision: 
            landing = bitshift(end_square, direction, 1)
            aggro_board[not self.player] ^= landing|collision
        return
    
    def all_legal_moves(self):
        '''Calculate the ending squares of legal moves in all directions and distances.'''
        moves = {}
        o1, p1 = self.boards[self.player]
        o2, p2 = self.boards[not self.player]
        for key, direction in DIRECTIONS.items():
            legals = self.legal_moves(direction, p1, p2, o1, o2)
            if legals:
                moves[key] = legals
        return moves

    @staticmethod
    def legal_moves(direction: int, bits_p1: int, bits_p2: int, bits_o1: int, bits_o2: int) -> dict[int,int]:
        'Calculate the ending squares of legal moves in a direction, given the bitboards of the current player and opponent.'
        legals = {}
        legal_passive = ~(bits_p1 | bits_o1)
        passive1 = bitshift(bits_p1, direction, 1) & legal_passive
        if passive1: # can play a passive move at distance 1
            legal_aggro = ~erode(bits_o2 | bits_p2, -direction, 1) & ~bits_p2
            aggro1 = bitshift(bits_p2, direction, 1) & legal_aggro
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
    game.move(1 << 5, DIRECTIONS['NW'],2)
    game.render()
    game.player = not game.player
    game.move(1 << 6, DIRECTIONS['SE'],1)
    game.render()
    pass