# Technical-Project-RL
University project to use reinforcement learning in games such as Tic-Tac-Toe, Connect-4 and Shōbu. These games and the reinforcement learning methods were coded from scratch. A bitboard implementation was used for Connect-4 and Shōbu. A one-step tabular minimax Q-learning method was applied to Tic-Tac-Toe. Whereas, an AlphaZero style algorithm was applied to Shōbu (reinforcement learning wasn't applied to Connect-4 due to time constraints).

## Running
There are two recommended methods of running the files within this repository, which were tested in MacOS but should work in other operating systems. Remember to install the imported python packages such as NumPy and TensorFlow etc.

### Command line
Use the python -m switch terminal command.

Example usage that assumes this repository is your current working directory (by using the unix cd command):

```
python -m shobu.array-based.logic3"
```

### VSCode

Alternatively, a VSCode launch.json file can be used to run and debug configurations of python modules.

(You may run some of the files with the VSCode run buttion but note that by default VSCode changes the working directory to the running file, meaning that files that import from parent directories encounter a ModuleNotFoundError. I was unable to find a more elegant solution to change this behaviour.)

Example VSCode launch.json:

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Module",
            "type": "python",
            "request": "launch",
            "module": "shobu.array-based.logic3",
            "justMyCode": true
        }
    ]
}
```

