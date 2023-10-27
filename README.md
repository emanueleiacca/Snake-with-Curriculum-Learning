## SnakeEnv Agent

This repository contains a SnakeEnv agent trained using Stable Baselines3. The agent was trained to play the game of Snake on a variety of board sizes, starting from small boards and gradually increasing in difficulty.

### Curriculum learning

The agent was trained using a curriculum learning approach, which means that it was first trained on simpler boards and then gradually moved on to more difficult boards. This helped the agent to learn the basics of the game before being challenged with more complex tasks.

### Performance

The agent is able to achieve high scores on a variety of board sizes, including the most difficult 20x20 board. It is also able to learn to play the game with different curriculum settings, which makes it a versatile agent that can be used in a variety of research settings.

### How to use the agent

To use the agent, simply clone this repository and install the required dependencies:

```bash
pip install stable-baselines3 gym snakeenv
```

Then, run the following command to start the agent:

```bash
python best_model.py
```

This will start the game of Snake and display the agent's playing strategy in real time
