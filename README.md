# Technical-Project-RL
University project to use Reinforcement Learning in games such as Tic-Tac-Toe, Connect-4 and Shōbu. 

## Running

I recommended using the python -m switch terminal command when running files that import from parent directories, shown below, otherwise ModuleNotFoundError with be encounted. (I experimented with modifying the python sys.path to do this automatically for users but to no success.)

Example command line usage:

```
python -m Shobu.logic"
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
            "module": "Shobu.logic",
            "justMyCode": true
        }
    ]
}
```
