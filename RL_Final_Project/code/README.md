# Reinforcment Learning Starter Files

Reinforcment learning starter files in order to immediatly train, visualize and evaluate an agent **without writing any line of code**.

<p align="center">
    <img width="300" src="README-rsrc/visualize-keycorridor.gif">
</p>

These files are suited for [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments and [`torch-ac`](https://github.com/lcswillems/torch-ac) RL algorithms. They are easy to adapt to other environments and RL algorithms.

## Features

- **Script to train**, including:
  - Log in txt, CSV and Tensorboard
  - Save model
  - Stop and restart training
  - Use A2C or PPO algorithms
- **Script to visualize**, including:
  - Act by sampling or argmax
  - Save as Gif
- **Script to evaluate**, including:
  - Act by sampling or argmax
  - List the worst performed episodes

## Example of use

Train, visualize and evaluate an agent on the `MiniGrid-DoorKey-5x5-v0` environment:

<p align="center"><img src="README-rsrc/doorkey.png"></p>

1. Train the agent on the `MiniGrid-DoorKey-5x5-v0` environment with PPO algorithm:

```
python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10 --frames 80000
```


2. Visualize agent's behavior:

```
python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

<p align="center"><img src="README-rsrc/visualize-doorkey.gif"></p>

3. Evaluate agent's performance:

```
python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

## Other examples

### Handle textual instructions

In the `GoToDoor` environment, the agent receives an image along with a textual instruction. To handle the latter, add `--text` to the command:

```
python3 -m scripts.train --algo ppo --env MiniGrid-GoToDoor-5x5-v0 --model GoToDoor --text --save-interval 10 --frames 1000000
```

<p align="center"><img src="README-rsrc/visualize-gotodoor.gif"></p>

### Add memory

In the `RedBlueDoors` environment, the agent has to open the red door then the blue one. To solve it efficiently, when it opens the red door, it has to remember it. To add memory to the agent, add `--recurrence X` to the command:

```
python3 -m scripts.train --algo ppo --env MiniGrid-RedBlueDoors-6x6-v0 --model RedBlueDoors --recurrence 4 --save-interval 10 --frames 1000000
```

<p align="center"><img src="README-rsrc/visualize-redbluedoors.gif"></p>

An example of use:

```
python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

<p align="center"><img src="README-rsrc/visualize-doorkey.gif"></p>

In this use case, the script displays how the model in `storage/DoorKey` behaves on the MiniGrid DoorKey environment.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--argmax`: select the action with highest probability
- ... (see more using `--help`)


An example of use:

```
python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

<p align="center"><img src="README-rsrc/evaluate-terminal-logs.png"></p>

<h2 id="model">model.py</h2>

The default model is discribed by the following schema:

<p align="center"><img src="README-rsrc/model.png"></p>

By default, the memory part (in red) and the langage part (in blue) are disabled. They can be enabled by setting to `True` the `use_memory` and `use_text` parameters of the model constructor.

This model can be easily adapted to your needs.
