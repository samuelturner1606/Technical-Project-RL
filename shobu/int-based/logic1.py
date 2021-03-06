'''
A fast and lightweight bitboard board game engine implementation of Shōbu.
'''
from random import choice, shuffle
from collections import deque
from dataclasses import dataclass

DIRECTIONS = {'N':6, 'E':-1, 'S':-6, 'W':1, 'NE':5, 'NW':7, 'SE':-7, 'SW':-5}
BITMASK = 0b1111001111001111001111 # bitmask deletes bits that move off the board
COMBOS:tuple[dict[str,tuple[tuple[int]]]] = ( # board indices: passive board (p,i) -> aggro boards (p,~i) or (~p,i)
    {'01':((0,0),(0,1)), '02':((0,0),(1,0)), '10':((0,1),(0,0)), '13':((0,1),(1,1))},
    {'20':((1,0),(0,0)), '23':((1,0),(1,1)), '31':((1,1),(0,1)), '32':((1,1),(1,0))} 
)

@dataclass
class State:
    'Uniquely defines a Shōbu game state.'
    
    boards:list[list[list[int]]] = [[[15,3932160], [15,3932160]],[[15,3932160], [15,3932160]]] # index 1 corresponds to player 1

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        print(f'-------- {str(message)}')
        for i in 0,1:
            left = self._mailbox(self.boards[i][0], i^0)
            right = self._mailbox(self.boards[i][1], i^1)
            for row in range(4):
                print(left[row]+right[row])
        return
    
    def is_terminal(self) -> bool:
        '''The first player to lose all their stones on any board loses the game. 
        Reward values 1 = player 1 win, -1 = player 2 win and 0 = game not over.
        Returns self.done'''
        if self.plies >= 1000: # Max game length avoids theoretical stalemates
            self.done = True
            self.reward = 0
            self.render('Max game length reached')
        elif not self.done: # if not already terminal from there being no legal moves or max game length
            for side in self.boards:
                for board in side:
                    if not board[0]:
                        self.done = True
                        self.reward = 1 # player 1 wins
                        break 
                    elif not board[1]:
                        self.done = True
                        self.reward = -1 # player 2 wins
                        break 
        return self.done 
    
    def make_move(self, passive_board: list[int], aggro_board: list[int], passive_end: int, aggro_end: int, direction: int, distance: int) -> None:
        'Update the two boards inplace with the legal passive and aggro moves.'
        passive_start = self._bitshift(passive_end, -direction, distance)
        passive_board[self.player] ^= (passive_start | passive_end)
        aggro_start = self._bitshift(aggro_end, -direction, distance)
        aggro_board[self.player] ^= (aggro_start | aggro_end)
        path = aggro_end | self._bitshift(aggro_end, -direction, 1)
        collision = path & aggro_board[not self.player]
        if collision: 
            landing = self._bitshift(aggro_end, direction, 1)
            aggro_board[not self.player] ^= (collision | landing)
        self.plies += 1
        self.player = not self.player
        return
    
    def all_legals(self) -> dict[tuple[tuple[int]],dict[str,dict[int,tuple]]]:
        '''Find all legal moves for all board combinations.\n
        Returns { boards: { direction : { distance : (passive moves, aggro moves) } } }\n
        Where boards are the indice tuples and moves are ending positions'''
        b = self.board_legals()
        output = {}
        for p, a in COMBOS[self.player]:
            passive_board = self.boards[p[0]][p[1]]
            aggro_board = self.boards[a[0]][a[1]]
            directions = {}
            for key in DIRECTIONS.keys():
                distances = {}
                passives = b[tuple(passive_board)][key][0]
                aggros = b[tuple(aggro_board)][key][1]
                if passives[0] and aggros[0]: # has legal moves at distance 1
                    distances[1] = (passives[0], aggros[0])
                    if passives[1] and aggros[1]: # has legal moves at distance 2
                        distances[2] = (passives[1], aggros[1])
                    directions[key] = distances
            if directions:
                output[(p, a)] = directions
        if not output: # if no legal moves
            self.render('No legal moves')
            self.done = True
            if self.player == 1: 
                
                self.reward = -1 # player 1 loses since they cannot move
            else:
                self.reward = 1
        return output
    
    def board_legals(self) -> dict[tuple,dict[int,tuple]]:
        '''Find all legal moves, in all directions, on all boards not including board combos.\n
        Returns { board : { direction : (passive moves, aggro moves) } }'''
        output = {}
        for side in self.boards:
            for board in side:
                if tuple(board) not in output:
                    temp = {}
                    for key, d in DIRECTIONS.items():
                        temp[key] = (self.legal_passives(board,d), self.legal_aggros(board,d))
                    output[tuple(board)] = temp
        return output

    def _legal_passives(self, passive_board: list[int], direction: int) -> tuple[int]:
        'Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board.'
        empty = ~(passive_board[self.player] | passive_board[not self.player])
        passive1 = self._bitshift(passive_board[self.player], direction, 1) & empty
        passive2 = self._bitshift(passive1, direction, 1) & empty
        return passive1, passive2
    
    def _legal_aggros(self, aggro_board: list[int], direction: int) -> tuple[int]:
        'Find the ending squares of all legal aggro moves, for all distances, in a direction, on a given board.'
        p2 = aggro_board[self.player]
        p3 = self._bitshift(p2, direction, 1)
        x2 = p2 | aggro_board[not self.player]
        x1 = self._bitshift(x2, -direction, 1)
        x3 = self._bitshift(x2, direction, 1)
        # not pushing more than one stone or your own stones
        legal_aggro1 = ~( (x1 & x2) | p2 )
        aggro1 = p3 & legal_aggro1 
        legal_aggro2 = ~( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) )
        aggro2 = self._bitshift(p2, direction, 2) & legal_aggro2
        return aggro1, aggro2

    def random_ply(self) -> None:
        'Randomly play a legal move. Faster than choosing from all legal moves.'
        #boards = self.boards.copy()
        combos = list(COMBOS[self.player])
        while combos:
            shuffle(combos)
            p, a = combos.pop()
            p_board = self.boards[p[0]][p[1]]
            a_board = self.boards[a[0]][a[1]]
            directions = list(DIRECTIONS.values())
            while directions:
                shuffle(directions)
                direc = directions.pop()
                p1, p2 = self.legal_passives(p_board, direc)
                a1, a2 = self.legal_aggros(a_board, direc)
                distances = [1,2]
                while distances:
                    shuffle(distances)
                    dist = distances.pop()
                    if dist == 1 and p1 and a1:
                        random_p = choice(self._split(p1))
                        random_a = choice(self._split(a1))
                        self.make_move(p_board, a_board, random_p, random_a, direc, dist)
                        return
                    elif dist == 2 and p2 and a2:
                        random_p = choice(self._split(p2))
                        random_a = choice(self._split(a2))
                        self.make_move(p_board, a_board, random_p, random_a, direc, dist)
                        return
        self.render('No legal random moves')
        self.done = True
        if self.player == 1: 
            self.reward = -1 # player 1 loses since they cannot move
        else:
            self.reward = 1
        self.player = not self.player
        return
    
    def human_ply(self) -> None:
        'Ask the user for a legal move and update the state with it.'
        legals = self.all_legals()
        if legals:
            combos = COMBOS[self.player]
            combo = combos[self._get_choice(list(combos), 'passive and aggro board combo')]
            p_board = self.boards[combo[0][0]][combo[0][1]]
            a_board = self.boards[combo[1][0]][combo[1][1]]
            direc = self._get_choice(list(legals[combo]), 'direction')
            dist = int(self._get_choice([ str(i) for i in list(legals[combo][direc]) ], 'distance'))
            passives, aggros = legals[combo][direc][dist]
            end1 = 1 << int(self._get_choice([str(n.bit_length()-1) for n in self._split(passives)],'passive move ending square'))
            end2 = 1 << int(self._get_choice([str(n.bit_length()-1) for n in self._split(aggros)],'aggressive move ending square'))
            self.make_move(p_board, a_board, end1, end2, DIRECTIONS[direc], dist)
            self.player = not self.player
        return

    @staticmethod
    def _bitshift(bits: int, direction: int, distance: int) -> int:
        shift = direction * distance
        if shift > 0:
            return (bits >> shift) & BITMASK
        else:
            return (bits << -shift) & BITMASK

    @staticmethod
    def _split(legal_moves: int) -> list[int]:
        'Split `legal moves` into invidual moves by the on-bits.'
        seperate_moves = []
        while legal_moves:
            seperate_moves.append(legal_moves & -legal_moves) # get least significant on-bit
            legal_moves &= legal_moves-1 # remove least significant on-bit
        return seperate_moves

    @staticmethod
    def _view(bits: int, message: str = '') -> None:
        'For debugging.'
        mailbox = [ bits & (1 << n) and 1  for n in range(24) ]
        mailbox_2D = [ mailbox[ 6*row : 6*row+6 ] for row in range(4) ]
        print(*mailbox_2D, sep='\n', end=f'\n{message}\n\n')
        return

    @staticmethod
    def _mailbox(bitboards: list[int], type: bool) -> list[list[int]]:
        'Transform the bitboards into a 2D mailbox representation. Type refers to whether the board is light = 0 or dark = 1.'
        if type:
            colour = '\033[34m' # blue
        else:
            colour = '\033[39m' # white
        mailbox_2D = 4 * [colour]
        for row in range(4):
            for n in range(6*row, 6*row+4):
                mailbox_2D[row] += str( (bitboards[1] & (1 << n) and 1) + 2 * (bitboards[0] & (1 << n) and 1) )
        return mailbox_2D

    @staticmethod
    def _get_choice(choices: list[str], prompt: str = ''):
        'General user input handling function.'
        choice = ''
        while choice not in choices:
            choice = input(f"Choose one {prompt} from [{', '.join(choices)}]:\n")
        return choice

    def count_actions(self):
        'Count the number of legal actions in the current state.'
        count = 0
        legals = self.all_legals()
        for combo in legals:
            for direction in legals[combo]:
                for distance in legals[combo][direction]:
                    p, a = legals[combo][direction][distance]
                    p_count = len(self._split(p))
                    a_count = len(self._split(a))
                    count += (p_count * a_count)
        return count

class Game:
    'Shōbu game.'
    def __init__(self) -> None:
        self.plies:int = 0
        self.done:bool = False
        self.path:deque = deque()
        reward:int = 0
        player:bool = True # player 1 = True, player 2 = False
        pass

if __name__ == '__main__':
    game = State()
    game.boards = [[15,565376],[15,565376]],[[15,565376],[15,565376]]
    game.render()