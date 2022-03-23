'''Generating statistics from random games.'''
import shobu as s
import small_shobu as sm
from time import perf_counter, sleep
import resource

def main():
    for _ in range(EPISODES):
        game = sm.Small_state()
        while True:
            #sleep(0.1)
            #game.render(game.player)
            num_actions.append(sm.count_actions(game))
            game.random_ply()
            if game.is_terminal():
                results.append(game.reward)
                game_lengths.append(game.plies)
                #game.render()
                #print(f'Player {game.reward % 3} won')
                break
    return
if __name__ == '__main__':
    num_actions = []
    results = []
    game_lengths = []
    EPISODES = 10000

    t1 = perf_counter()
    main()
    print(f'Memory usage (MB): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000}')
    t2 = perf_counter()

    print(f'Time per game: {(t2-t1)/EPISODES}')
    print(f'Player 1 win-rate {results.count(1)/len(results)}, draw-rate {results.count(0)/len(results)} and loss-rate {results.count(-1)/len(results)}')

    print(f'Min game length: {min(game_lengths)}')
    print(f'Max game length: {max(game_lengths)}')
    print(f'Mean game length: {sum(game_lengths)/len(game_lengths)}')

    print(f'Min number of actions: {min(num_actions)}')
    print(f'Max number of actions: {max(num_actions)}')
    print(f'Mean number of actions: {sum(num_actions)/len(num_actions)}')



