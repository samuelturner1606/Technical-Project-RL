# Technical-Project-RL
University project to use Reinforcement Learning in games such as Tic-Tac-Toe, Connect-4 and Shōbu. 

## Running
It is recommended to use the python -m switch terminal command when running files that import from parent directories, otherwise ModuleNotFoundError may be encounted (I was unable to implement other solutions.)

Example command line usage assuming the repo is your current working directory:

```
python -m shobu.small.logic"
```

Alternatively to accomplish this, a VSCode launch.json file can be used to run and debug configurations of python modules.

Example VSCode launch.json:

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Module",
            "type": "python",
            "request": "launch",
            "module": "shobu.small.logic",
            "justMyCode": true
        }
    ]
}
```
