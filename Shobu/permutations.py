'''Calculating state-space complexity.'''
from itertools import product
from math import log10

perms_with_repeats = list(product([0,1,2],repeat=16))
print(f'Upper bound: {len(perms_with_repeats)}')
constraint = [] # remove permutations where there more than 4 white or black stones
terminal = [] # permutations that are terminal boards
for i, p in enumerate(perms_with_repeats):
    c1 =  p.count(1)
    c2 = p.count(2)
    if c1 <= 4 and c2 <= 4:
        constraint.append(p)
    if c1 == 0 or c2 == 0:
        terminal.append(p)
constraint.pop(0) # remove permutation with zero stone which is illegal
l = len(constraint)
t = len(terminal)
print(f'Legal permutations: {l}')
print(f'Terminal boards: {t}')
s = l * (l-t)**3
print(f'State-space complexity: {s}, log10 is {log10(s)}') # permutations of 4 boards
pass