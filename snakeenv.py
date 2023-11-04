import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
#This environment simulates the classic Snake game, where a snake tries to eat apples to grow longer while avoiding collisions with boundaries and itself.
#we have 4 different action(representing the direction the snake should move), and an observation space consisting of various parameters related to the state of the game

SNAKE_LEN_GOAL = 30
tableSizeObs = 500
tableSize = 400   ##!!
halfTable = int(tableSize/2)

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,tableSize/10)*10,random.randrange(1,tableSize/10)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=tableSize or snake_head[0]<0 or snake_head[1]>=tableSize or snake_head[1]<0 :
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnv(gym.Env):
    width = 400
    height = 400
    curriculum = 4

    def __init__(self,height,width,curriculum):
        super(SnakeEnv, self).__init__()
        self.apple_reward = 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-tableSizeObs, high=tableSizeObs,
                                            shape=(5+SNAKE_LEN_GOAL,), dtype=np.float64)

    def step(self, action):
        self.prev_actions.append(action)
        self.img = np.zeros((tableSize,tableSize,3),dtype='uint8')
        self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)

        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10


        apple_reward = 10000
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            self.apple_reward = apple_reward
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()

    # Increase the reward for eating multiple apples in a row
        if self.apple_reward > 0:
            self.apple_reward *= 1.1

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()

        self_eating_penalty = -300
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            self.done = True
            self.reward = self_eating_penalty


        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.total_reward = ((halfTable - euclidean_dist_to_apple) + apple_reward)/100

        #print(self.total_reward)

        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.reward

        if self.done:
            self.reward = -300
        info = {}


        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.total_reward, self.done, info
    
    def reset(self):
        self.img = np.zeros((tableSize,tableSize,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[halfTable,halfTable],[halfTable - 10,halfTable],[halfTable - 20,halfTable]]
        self.apple_position = [random.randrange(1,tableSize/10) * 10,random.randrange(1,tableSize/10) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [halfTable,halfTable]

        self.prev_reward = 0
        self.total_reward = 0
        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1) # to create history

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation
