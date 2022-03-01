DIRECTIONS = {'N':5, 'E':-1, 'S':-5, 'W':1, 'NE':4, 'NW': 6, 'SE': -6, 'SW': -4}
BITMASK = 0b01111011110111101111
player = True # player 1 = True, player 2 = False

class Board:
    '''Bitboard representation of one of the game boards.'''
    def __init__(self) -> None:
        self.bitboards = [0b1111, 0b01111000000000000000] # index 1 corresponds to the bitboard of player 1

    def mailbox(self) -> list[list[int]]:
        'Transform two bitboards into a 2D mailbox representation and print it.'
        mailbox = [ (self.bitboards[1] & (1 << n) and 1) + 2*(self.bitboards[0] & (1 << n) and 1) for n in range(20) ]
        mailbox_2D = [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]
        print(*mailbox_2D, sep='\n', end='\n\n')
        return mailbox_2D
    
    def __hash__(self) -> int:
        return self.bitboards[1] | self.bitboards[0]

class Shobu:
    def __init__(self) -> None:
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
    
    def passive_move(self, homeboard: Board, start_square: int, direction: int, distance: int) -> None:
        'Updates a board with a passive move, assuming it is legal.'
        end_square =  self.bitshift(start_square, direction, distance)
        homeboard.bitboards[player] ^= start_square|end_square
        return
    
    def legal_passives(self, bits: int, direction: int, distance: int, empty: int):
        output = {1:{}, 2:{}}
        for dist in output.keys:
            for d in DIRECTIONS.keys:
                nbors = self.bitshift(bits, d, dist) & empty
                
        return output
    
    @staticmethod
    def bitshift(bits: int, direction: int, distance: int) -> int:
        shift = direction * distance
        if shift > 0:
            return bits >> shift
        else:
            return (bits << -shift) & BITMASK # bitmask deletes bits that move off the board

s = Shobu()
s.render()
s.passive_move(s.boards[1], 0b0100000000000000000, DIRECTIONS['NW'], 2)
s.render()