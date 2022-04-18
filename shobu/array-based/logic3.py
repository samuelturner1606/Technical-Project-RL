'''
Board game CLI implementation of Shōbu using bitwise operators.
'''
from random import choice
import numpy as np
import colorama

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
    @param boards: 2x8x8 np.ndarray'''
    # The following are constants shared between all State instances used to speed up calculations
    P2A =  { 2: (0, 3), 3: (1, 2) } # maps passive to aggro boards of current player when orientated to them
    A2P = { 0:2, 3:2, 1:3, 2:3 } # reverse mapping of `P2A`
    DIRECTIONS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    OFFSETS = ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)) # indices match `DIRECTIONS`
    QUADRANT = ( # Used to slice an nx8x8 ndarray into nx4x4, via the board indices 0 to 3.
        (..., slice(0,4), slice(0,4)),
        (..., slice(0,4), slice(4,8)),
        (..., slice(4,8), slice(0,4)),
        (..., slice(4,8), slice(4,8)) )

    def __init__(self, boards: np.ndarray = None) -> None:
        if boards is None: # creates default starting state
            self.boards: np.ndarray = np.zeros(shape=(2,8,8), dtype=np.uint8)
            self.boards[0, -1:0:-4, :] = 1 # player's stones
            self.boards[1, 0:-1:4, :] = 1 # opponent's stones
        else:
            self.boards: np.ndarray = np.asarray(boards, dtype=np.uint8)
            assert self.boards.shape == (2,8,8), 'Board not the right shape.'
    
    def __eq__ (self, other):
        if isinstance(other, State):
            return np.all(self.boards == other.boards)
        else:
            return NotImplementedError 

    @staticmethod
    def _shift(array: np.ndarray, offset: tuple[int]) -> np.ndarray:
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
        @return passive1: 2x4x4 ndarray with a legal passive move, of distance 1, applied to all stones of the current player
        @return passive2: 2x4x4 ndarray with a legal passive move, of distance 2, applied to all stones of the current player'''
        assert p_board.shape == (2,4,4), 'Passive board not the right shape.'
        player = p_board[0]
        empty = 1 - (player | p_board[1])
        passive1 = self._shift(player, offset) & empty
        passive2 = self._shift(passive1, offset) & empty
        return passive1, passive2
    
    def _legal_aggros(self, a_board: np.ndarray, offset: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal aggressive moves, for all distances, in a direction, on a given board.
        @param a_board: A 2x4x4 ndarray representing the board to play a aggressive move on.
        @param offset: Directional offset
        @return aggro1: A 2x4x4 ndarray with a legal aggressive move, of distance 1, applied to all stones of the current player
        @return aggro2: A 2x4x4 ndarray with a legal aggressive move, of distance 2, applied to all stones of the current player'''
        assert a_board.shape == (2,4,4), 'Aggressive board not the right shape.'
        p2 = a_board[0]
        p3 = self._shift(p2, offset)
        x2 = p2 | a_board[1]
        x1 = self._shift(x2, tuple(map((-1).__mul__, offset)))
        x3 = self._shift(x2, offset)
        mask1 = 1 - ( (x1 & x2) | p2 ) 
        aggro1 = p3 & mask1 
        mask2 = 1 - ( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) ) # not pushing more than one stone or your own stones
        aggro2 = self._shift(p3, offset) & mask2
        return aggro1, aggro2

    def legal_actions(self) -> np.ndarray:
        '''Find all legal actions on all boards, in all directions and distances. Can be used to mask illegal actions from the policy head of the neural network.
        @return output: ndarray indexed by (combo, direction, distance, aggro_y, aggro_x, passive_y, passive_x)'''
        output = np.zeros(shape=(4,8,2,4,4,4,4), dtype=np.uint8)
        for d, offset in enumerate(State.OFFSETS):
            for p in State.P2A:
                passives1, passives2 = self._legal_passives(self.boards[State.QUADRANT[p]], offset)
                if np.any(passives1):
                    for a in State.P2A[p]:
                        aggros1, aggros2 = self._legal_aggros(self.boards[State.QUADRANT[a]], offset)
                        temp_aggros = np.zeros(shape=(4,8,2,4,4), dtype=np.uint8)
                        if np.any(aggros1):
                            output[a, d, 0, ...] = passives1
                            temp_aggros[a, d, 0, ...] = aggros1
                        if np.any(aggros2) and np.any(passives2):
                            output[a, d, 1, ...] = passives2
                            temp_aggros[a, d, 1, ...] = aggros2
                        output[a, d, ...] &= temp_aggros[a, d, ..., None, None] # temp_aggros is broadcasted to the shape of output
        assert output.shape == (4,8,2,4,4,4,4), 'Shape of legal actions is wrong'
        return output
    
    def apply(self, action: tuple[int]) -> np.ndarray:
        '''Returns a copy of the boards with (an assumed) legal passive and aggressive move having been applied.
        @return new_boards: 2x8x8 ndarray'''
        assert len(action) == 7, 'Action not the right shape.'
        a, direction, distance, a_y, a_x, p_y, p_x = action
        new_boards = self.boards.copy()
        offset = State.OFFSETS[direction]
        neg_offset = tuple(map((-(distance+1)).__mul__, offset))
        # construct boards from coords
        p_end = np.zeros((4,4), new_boards.dtype)
        a_end = p_end.copy()
        p_end[p_y, p_x] = 1
        a_end[a_y, a_x] = 1
        p = State.A2P[a]
        p_board = new_boards[State.QUADRANT[p]]
        a_board = new_boards[State.QUADRANT[a]]
        # update current player's stones
        p_board[0] ^= (self._shift(p_end, neg_offset) | p_end)
        a_board[0] ^= (self._shift(a_end, neg_offset) | a_end)
        # detect collision with opponent's stones
        path = a_end | self._shift(a_end, tuple(map((-1).__mul__, offset)) )
        collision = path & a_board[1]
        print(f'combo: {p, a}, direction: {State.DIRECTIONS[direction]}, distance: {distance+1}, p: {p_x, p_y}, a: {a_x, a_y}')
        if np.any(collision): 
            print('collision with opponent stone')
            landing = self._shift(a_end, offset)
            a_board[1] ^= (collision | landing)
        assert new_boards.shape == (2,8,8)
        return new_boards[::-1,::-1,::-1] # orientates boards to next player by flipping player planes and rotating them 180 degrees

    def render(self) -> None:
        'Prints the current game boards.'
        m = (self.boards[0] + 2*self.boards[1]).astype(str)
        m = np.insert(m, 4, 8*['|'], 1)
        m = np.insert(m, 4, 9*['-'], 0)
        txt: str = '\n ' + np.array_str(m)
        txt = txt.replace('\'','')
        txt = txt.replace('[','')
        txt = txt.replace(']','')
        txt = txt.replace('0', colorama.Fore.WHITE+'o')
        txt = txt.replace('1', colorama.Fore.BLUE+'o')
        txt = txt.replace('2', colorama.Fore.RED+'o')
        txt = txt.replace('|',colorama.Fore.GREEN+'|')
        txt = txt.replace('-',colorama.Fore.GREEN+'-')
        txt += colorama.Style.RESET_ALL
        out = txt.splitlines(True)
        return print(*out)

class Human:
    def policy(self, legal_actions: np.ndarray) -> tuple[int]:
        '''Select a passive and aggressive action from all legal actions.'''
        def get_choice(choices: list[str], prompt: str = '') -> str:
            'General user input handling function.'
            if len(choices) < 2:
                choice = choices[0]
                print(f'{prompt} of {choice} was the only choice left.')
            else:
                choice = ''
                while choice not in choices:
                    choice = input(f"Choose one {prompt} from [{', '.join(choices)}]:\n")
            return choice
        action = np.argwhere(legal_actions)
        # filter combos
        combos = action[:, 0]
        unique_combos = [str(i) for i in np.unique(combos)]
        combo = int(get_choice(unique_combos, 'aggro board'))
        action = action[combos == combo]
        # filter directions
        directions = action[:, 1]
        unique_directions = [State.DIRECTIONS[i] for i in np.unique(directions)]
        direction = State.DIRECTIONS.index(get_choice(unique_directions, 'direction'))
        action = action[directions == direction]
        # filter distances
        distances = action[:, 2]
        unique_distances = [str(i+1) for i in np.unique(distances)]
        distance = unique_distances.index(get_choice(unique_distances, 'distance'))
        action = action[distances == distance]
        # filter passives
        passives = action[:, 5:7]
        unique_passives = [ f'({x} {y})' for y, x in np.unique(passives, axis=0)]
        p = get_choice(unique_passives, 'Passive (x y)')
        passive = np.array([[p[3],p[1]]], dtype=action.dtype)
        action = action[np.all(passives == passive, axis=1)]
        # filter aggros
        aggros = action[:, 3:5]
        unique_aggros = [ f'({x} {y})' for y, x in np.unique(aggros, axis=0)]
        a = get_choice(unique_aggros, 'Aggro (x y)')
        aggro = np.array([[a[3],a[1]]], dtype=action.dtype)
        action = action[np.all(aggros == aggro, axis=1)]
        return tuple(action[0])

class Random:
    def policy(self, legal_actions: np.ndarray) -> tuple[int]:
        '''Randomly select an action from all legal actions.'''
        return tuple(choice(np.argwhere(legal_actions))) 

class Game:
    def __init__(self, player1: object, player2: object, render: bool = False) -> None:
        self.players = [player1, player2]
        self.render = render
        if render:
            colorama.init()
            print(colorama.ansi.clear_screen())
        self.reset()

    def reset(self) -> None:
        self.state = State() # default start state
        self.state_history = []
        self.action_history = []
        return
    
    @property
    def current_player(self):
        return len(self.action_history) % 2
    
    @property
    def plies(self):
        return len(self.action_history)

    def terminal(self, legal_actions: np.ndarray):
        '''Losing conditions:
        1) A player has no stones on one of the boards
        2) A player has no legal moves
        
        Drawing conditions:
        1) The game has lasted 150 plies

        @return reward: reward value for player 1
        - win = `1`
        - draw = `0.5`
        - loss = `0`
        - non-terminal = `None`
        ### Note
        Checking for terminal conditions is done just before a player makes their move in case they have no legal moves.'''
        if self.plies >= 150:
            return 0.5 
        elif (not np.any(legal_actions)):
            return self.current_player
        for i in 0,1,2,3:
            board = self.state.boards[State.QUADRANT[i]][0]
            if not np.any(board): 
                return self.current_player # opponent won on previous turn
        return None  # game not over

    def play(self) -> int:
        '''Plays a Shōbu game then resets itself.
        @return state_history: list of game board states
        @return action_history: list of game actions
        @return reward_history: list of length plies where elements are the terminal reward from perspective of the player at that time
        #### Example `reward_history`:
        - If player `1` won in `5` plies then `reward_history` = [1, 0, 1, 0, 1]
        - Whereas, if player `2` won in `4` plies then `reward_history` = [0, 1, 0, 1]

        This is done so that the neural network critic can identify the value of states based on the future reward. 
        The critic value is from the perspective of the current player's state so that 0 means it thinks the opponent will win. 
        This is `NOT` always the same as saying player 2 will win as it depends on who's turn it is.
        '''
        while True:
            legal_actions = self.state.legal_actions()
            if self.render:
                self.state.render()
                n = np.argwhere(legal_actions)
                print(f'Number of legal actions: {len(n)}')
            reward = self.terminal(legal_actions)
            if reward is not None:
                print(f'plies: {self.plies}, reward: {reward}')
                s = self.state_history
                a = self.action_history
                r = (self.plies//2)*[reward, 1-reward] + (self.plies%2)*[reward]
                assert len(s) == len(a)
                self.reset()
                return s, a, r
            action = self.players[self.current_player].policy(legal_actions)
            self.state_history.append(self.state.boards.astype(np.float32))
            self.action_history.append(action)
            new_boards = self.state.apply(action)
            self.state = State(new_boards)

if __name__ == '__main__':
    game = Game(Random(), Random(), True)
    s, a, r = game.play()
    pass