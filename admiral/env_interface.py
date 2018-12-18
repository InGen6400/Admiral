import gym
import gym.spaces
import numpy as np


class SeaGameJava(gym.core.Env):

    def __init__(self):
        self.action_space = gym.spaces.Discrete(25)
