'''
Board game CLI implementation of Shōbu using bitwise operators.
'''
from random import choice
import numpy as np

'''Constants used for testing:'''
MAX_ACTIONS = [
    [[1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]],
    
    [[0,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0],
    [0,1,0,1,0,1,0,1],
    [0,1,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0],
    [0,1,0,1,0,1,0,1],
    [0,1,0,0,0,1,0,0]]
    ]
STALEMATE = [
    [[0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,1,1]],
    
    [[1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]]
    ]
NO_MOVES = [
        [[0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,1,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,1,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0]],
        
        [[0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0]]
        ]

class State:
    '''4 Shōbu 4x4 boards.
    @param boards: 2x8x8 np.ndarray or None
    @param current_player: 1 for player1, 0 for player2.'''
    # The following are constants shared between all State instances used to speed up calculations
    P2A = ( # mapping of passive to aggro board (indices) combinations
        { 0: (1, 2), 1: (0, 3)}, # player 2
        { 2: (0, 3), 3: (1, 2)} )# player 1
    A2P = tuple( {aggro:passive for passive in player for aggro in player[passive]} for player in P2A ) # reverse mapping of `P2A`
    DIRECTIONS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    OFFSETS = ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)) # indices match `DIRECTIONS`
    QUADRANT = ( # Used to slice an nx8x8 ndarray into nx4x4, via the board indices 0 to 3.
        (..., slice(0,4), slice(0,4)),
        (..., slice(0,4), slice(4,8)),
        (..., slice(4,8), slice(0,4)),
        (..., slice(4,8), slice(4,8)) )

    def __init__(self, boards: np.ndarray = None, current_player: int = 1) -> None:
        assert current_player in (0,1), 'Player parameter incorrect.'
        self.current_player = current_player
        if boards is None: # creates default starting state
            self.boards: np.ndarray = np.zeros(shape=(2,8,8), dtype=np.uint8)
            self.boards[1, -1:0:-4, :] = 1 # player 1 pieces
            self.boards[0, 0:-1:4, :] = 1 # player 2 pieces
        else:
            self.boards: np.ndarray = np.asarray(boards, dtype=np.uint8)
            assert self.boards.shape == (2,8,8), 'Board not the right shape.'
    
    def __eq__ (self, other):
        if isinstance(other, State):
            return (self.current_player == other.current_player) and (self.boards == other.boards).all()
        else:
            return NotImplementedError 

    def _shift(self, array: np.ndarray, offset: tuple[int]) -> np.ndarray:
        'Returns a copy of an `array` shifted (without rollover) by `offset`.'
        assert len(offset) == array.ndim, 'Number of dimensions do not match.'
        new_array = np.empty_like(array)
        def slice1(o):
            return slice(o, None) if o >= 0 else slice(0, o)
        new_array[tuple(slice1(o) for o in offset)] = (array[tuple(slice1(-o) for o in offset)])
        for axis, o in enumerate(offset):
            new_array[(slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)] = 0
        return new_array

    def _legal_passives(self, p_board: np.ndarray, offset: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board.
        @param p_board: 2x4x4 ndarray representing the board to play a passive move on.
        @param offset: directional offset
        @return passive1: 2x4x4 ndarray with a legal passive move, of distance 1, applied to all pieces of the current player
        @return passive2: 2x4x4 ndarray with a legal passive move, of distance 2, applied to all pieces of the current player'''
        assert p_board.shape == (2,4,4), 'Passive board not the right shape.'
        player = p_board[self.current_player]
        opponent = p_board[1 - self.current_player]
        empty = 1 - (player | opponent)
        passive1 = self._shift(player, offset) & empty
        passive2 = self._shift(passive1, offset) & empty
        return passive1, passive2
    
    def _legal_aggros(self, a_board: np.ndarray, offset: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal aggressive moves, for all distances, in a direction, on a given board.
        @param a_board: A 2x4x4 ndarray representing the board to play a aggressive move on.
        @param offset: Directional offset
        @return aggro1: A 2x4x4 ndarray with a legal aggressive move, of distance 1, applied to all pieces of the current player
        @return aggro2: A 2x4x4 ndarray with a legal aggressive move, of distance 2, applied to all pieces of the current player'''
        assert a_board.shape == (2,4,4), 'Aggressive board not the right shape.'
        p2 = a_board[self.current_player]
        p3 = self._shift(p2, offset)
        x2 = p2 | a_board[1 - self.current_player]
        x1 = self._shift(x2, tuple(map((-1).__mul__, offset)))
        x3 = self._shift(x2, offset)
        mask1 = 1 - ( (x1 & x2) | p2 ) 
        aggro1 = p3 & mask1 
        mask2 = 1 - ( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) ) # not pushing more than one stone or your own stones
        aggro2 = self._shift(p3, offset) & mask2
        return aggro1, aggro2

    def legal_actions(self) -> np.ndarray:
        '''Find all legal actions on all boards, in all directions and distances. Can be used to directly mask the policy head of the neural network for legal actions.
        @return output: ndarray indexed by (direction, combo, distance, aggro_y, aggro_x, passive_y, passive_x)'''
        output = np.zeros(shape=(8,4,2,4,4,4,4), dtype=np.uint8)
        for d, offset in enumerate(State.OFFSETS):
            for p in State.P2A[self.current_player]:
                passives1, passives2 = self._legal_passives(self.boards[State.QUADRANT[p]], offset)
                if passives1.any():
                    for a in State.P2A[self.current_player][p]:
                        aggros1, aggros2 = self._legal_aggros(self.boards[State.QUADRANT[a]], offset)
                        temp_aggros = np.zeros(shape=(8,4,2,4,4), dtype=np.uint8)
                        if aggros1.any():
                            output[d, a, 0, ...] = passives1
                            temp_aggros[d, a, 0, ...] = aggros1
                        if aggros2.any() and passives2.any():
                            output[d, a, 1, ...] = passives2
                            temp_aggros[d, a, 1, ...] = aggros2
                        output[d, a, ...] &= temp_aggros[d, a, ..., None, None] # temp_aggros is broadcasted to the shape of output
        n = np.argwhere(output)
        print(f'Number of actions: {len(n)}')
        assert output.shape == (8,4,2,4,4,4,4), 'Shape of legal actions is wrong'
        return output
    
    def play_action(self, action: tuple[int]) -> np.ndarray:
        '''Returns a copy of the boards with (an assumed) legal passive and aggressive move having been applied.
        @return new_boards: 2x8x8 ndarray'''
        assert len(action) == 7
        direction, a, distance, a_y, a_x, p_y, p_x = action
        new_boards = self.boards.copy()
        offset = State.OFFSETS[direction]
        neg_offset = tuple(map((-(distance+1)).__mul__, offset))
        
        p_end = np.zeros((4,4), new_boards.dtype)
        a_end = p_end.copy()
        p_end[p_y, p_x] = 1
        a_end[a_y, a_x] = 1

        p = State.A2P[self.current_player][a]
        p_board = new_boards[State.QUADRANT[p]]
        a_board = new_boards[State.QUADRANT[a]]

        p_board[self.current_player] ^= (self._shift(p_end, neg_offset) | p_end)
        a_board[self.current_player] ^= (self._shift(a_end, neg_offset) | a_end)

        path = a_end | self._shift(a_end, tuple(map((-1).__mul__, offset)) )
        collision = path & a_board[1 - self.current_player]
        print(f'player: {self.current_player}, direction: {State.DIRECTIONS[direction]}, distance:{distance+1}, combo: {p, a}, p: {p_x, p_y}, a:{a_x, a_y}')
        if collision.any(): 
            print('collision')
            landing = self._shift(a_end, offset)
            a_board[1 - self.current_player] ^= (collision | landing)
        assert new_boards.shape == (2,8,8)
        return new_boards

        

    def render(self) -> None:
        'Prints the current game boards.'
        m = (2*self.boards[0] + self.boards[1]).astype(str)
        m = np.insert(m, 4, 8*['|'], 1)
        m = np.insert(m, 4, 9*['-'], 0)
        txt: str = '\n ' + np.array_str(m)
        txt = txt.replace('\'','')
        txt = txt.replace('[','')
        txt = txt.replace(']','')
        out = txt.splitlines(True)
        return print(*out)

class Human:
    def policy(self, legal_actions: np.ndarray, current_player: bool) -> tuple[int, int, int, int, int, int]:
        '''Select a passive and aggressive action from all legal actions.
        @return z1, y1, x1, z2, y2, x2: passive and aggressive actions defined using coordinates'''
        raise NotImplementedError
        def get_choice(choices: list[str], prompt: str = '') -> str:
            'General user input handling function.'
            choice = ''
            while choice not in choices:
                choice = input(f"Choose one {prompt} from [{', '.join(choices)}]:\n")
            return choice
        A = np.split(legal_actions, 8, axis=0) # split legal actions by direction
        q = State.DIRECTIONS.index( get_choice([State.DIRECTIONS[i] for i in range(8) if A[i].any()], 'direction') ) # choose a direction
        B = A[q] 
        w = int(get_choice([ str(i-1) for i in [2,3] if B[i].any() ], 'distance') ) + 1 # choose a distance
        z2 = 4*q + w
        C = np.argwhere(B[w])
        a_coord = get_choice([ f'({x} {y})' for y, x in C], 'aggro move (x y) ending square') # choose aggro ending
        x2 = int(a_coord[1]); y2 = int(a_coord[3])
        z1 = z2 - 2
        p_options = legal_actions[z1]
        p_mask = np.zeros(p_options.shape, p_options.dtype)[None,:]
        p_board = State.A2P[current_player][x2//4 + 2*(y2//4)] 
        p_mask[State.QUADRANT[p_board]] = 1
        p_options &= p_mask[0]
        D = np.argwhere(p_options)
        p_coord = get_choice([ f'({x} {y})' for y, x in D], 'passive move (x y) ending square') # choose passive ending
        x1 = int(p_coord[1]); y1 = int(p_coord[3])
        return z1, y1, x1, z2, y2, x2

class Random:
    def policy(self, legal_actions: np.ndarray, _) -> tuple[int, int, int, int, int, int]:
        '''Randomly select an action from all legal actions.'''
        return choice(np.argwhere(legal_actions).tolist()) 

class Game:
    def __init__(self, player1: object, player2: object, render: bool = False) -> None:
        self.players = [player2, player1] # index 1 for player 1, matching State.current_player
        self.render = render
        self._reset()

    def _reset(self) -> None:
        self.state = State() # default start state
        self.plies = 0 # used for max game length rule
        self.done = False
        return

    def _is_terminal(self, legal_actions: np.ndarray):
        '''Losing conditions:
        1) A player has no stones on one of the boards
        2) A player has no legal moves
        
        Drawing conditions:
        1) The game has lasted 150 plies

        @return reward: 
        - player1 win = `1`
        - draw = `0.5`
        - player1 loss = `0`
        - non-terminal = `None`
        ### Note
        Checking for terminal conditions is done just before a player makes their move in case they have no legal moves.'''
        if self.plies >= 150:
            self.done = True
            return 0.5
        elif (not legal_actions.any()):
            self.done = True
            return 1 - self.state.current_player
        for i in 0,1,2,3:
            board = self.state.boards[State.QUADRANT[i]][self.state.current_player]
            if not board.any():
                self.done = True
                return 1 - self.state.current_player # opponent won on previous turn
        return None  # game not over

    def play(self) -> int:
        'Plays a Shōbu game, then resets itself and returns the reward.'
        while True:
            if self.render:
                self.state.render()
            legal_actions = self.state.legal_actions()
            reward = self._is_terminal(legal_actions)
            if self.done:
                print(f'plies: {self.plies}, reward: {reward}')
                self._reset()
                return reward
            c = self.state.current_player
            action = self.players[c].policy(legal_actions, c)
            new_boards = self.state.play_action(action)
            self.state = State(new_boards, current_player=1-c)
            self.plies += 1

if __name__ == '__main__':
    game = Game(Random(), Random(), True)
    for _ in range(20):
        reward = game.play()
    pass