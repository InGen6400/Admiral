import gym
import gym.spaces
from enum import Enum, auto
import numpy as np
import socket

HOST = 'localhost'
PORT = 10000
BUFFER_SIZE = 4096

NONE = ''
UP = 'up\n'.encode()
DOWN = 'down\n'.encode()
RIGHT = 'right\n'.encode()
LEFT = 'left\n'.encode()
STAT = 'stat\n'.encode()

Commands = [NONE, UP, DOWN, RIGHT, LEFT, STAT]


class SeaGameJava(gym.core.Env):
    def __init__(self, name):
        # 4方向+停止が2回分のアクション
        self.action_space = gym.spaces.Box(
            np.array([0, 0]),
            np.array([5, 5]),
            dtype=np.uint8
        )

        # 自分のpoint, 各座標の相手のポイント, 各座標のタンクのポイント
        self.observation_space = gym.spaces.Dict({
            "myPoint": gym.spaces.Box(low=0, high=1200, dtype=np.uint16),
            "eShip": gym.spaces.Box(0, 1200, shape=[256, 256, 1], dtype=np.uint16),
            "tank": gym.spaces.Box(0, 4, shape=[256, 256, 1], dtype=np.uint8)
        })

        # ソケット確保
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))

        # ログインする
        self.name = name
        data = 'login ' + name + '\n'
        self.sock.sendall(data.encode())

    def step(self, action):
        self.sock.sendall(Commands[action[0]])
        self.sock.sendall(Commands[action[1]])
        
        return 0

    def reset(self):
        return 0
