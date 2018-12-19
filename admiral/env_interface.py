from typing import List

import gym
import gym.spaces
from enum import Enum, auto
import numpy as np
import socket

from numpy.core.multiarray import ndarray

HOST = 'localhost'
PORT = 10000
BUFFER_SIZE = 8192

NONE = ''.encode()
UP = 'up\n'.encode()
DOWN = 'down\n'.encode()
RIGHT = 'right\n'.encode()
LEFT = 'left\n'.encode()
STAT = 'stat\n'.encode()
RESET = 'reset\n'.encode()

Commands = [NONE, UP, DOWN, RIGHT, LEFT, STAT]


class SeaGameJava(gym.core.Env):
    name: str
    tank_map: ndarray
    ship_map: ndarray

    def __init__(self, name):
        # 4方向+停止が2回分のアクション
        self.action_space = gym.spaces.Box(
            np.array([0, 0]),
            np.array([5, 5]),
            dtype=np.uint8
        )

        # 自分のpoint, 各座標の相手のポイント, 各座標のタンクのポイント
        self.observation_space = gym.spaces.Dict({
            "myPoint": gym.spaces.Box(low=0, high=1200, shape=(1, ), dtype=np.uint16),
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

        self.ship_map = np.zeros((256, 256))
        self.tank_map = np.zeros((256, 256))
        self.my_x = 0
        self.my_y = 0
        self.my_point = 0
        self.finished = False

    def step(self, action):
        self.sock.sendall(Commands[action[0]])
        self.sock.sendall(Commands[action[1]])

        self.sock.sendall(STAT)

        sum_enemy_point = self.reload()

        return self.observe(), self.my_point - sum_enemy_point, self.finished

    def observe(self):
        observation = {
            'myPoint': self.my_point,
            'eShip': self.ship_map,
            'tank': self.tank_map
        }
        return observation

    def reset(self):

        self.sock.sendall(RESET)

        self.my_x = 0
        self.my_y = 0
        self.my_point = 0
        self.finished = False

        self.reload()

        return self.observe()

    # サーバーから情報を取得して状態を更新する
    def reload(self):
        recv: bytes = self.sock.recv(BUFFER_SIZE)
        lines = recv.decode().split('\n')
        line_num = 0
        # 読み飛ばし
        while lines[line_num] != "ship_info":
            if lines[line_num] == "finished":
                self.finished = True
            line_num = line_num + 1

        line_num = line_num + 1
        self.ship_map.fill(0)
        sum_enemy_point = 0
        # 敵の船情報の取得
        while lines[line_num] != ".":
            args: List[str] = lines[line_num].split(' ')
            name = args[0]
            x = int(args[1])
            y = int(args[2])
            point = int(args[3])
            # 自分以外
            if name == self.name:
                self.ship_map[x][y] = point
                sum_enemy_point = sum_enemy_point + point
            else:
                self.my_x = x
                self.my_y = y
                self.my_point = point
            line_num = line_num + 1

        # 読み飛ばし
        while lines[line_num] != "energy_info":
            line_num = line_num + 1

        line_num = line_num + 1
        self.tank_map.fill(0)
        # タンク情報の取得
        while lines[line_num] != ".":
            args: List[str] = lines[line_num].split(' ')
            x = int(args[0])
            y = int(args[1])
            point = int(args[2])

            self.tank_map[x][y] = point
            line_num = line_num + 1

        # 自分を中心にする
        self.ship_map = np.roll(self.ship_map, 128-self.my_x, axis=1)
        self.ship_map = np.roll(self.ship_map, 128-self.my_y, axis=0)
        self.tank_map = np.roll(self.tank_map, 128-self.my_x, axis=1)
        self.tank_map = np.roll(self.tank_map, 128-self.my_y, axis=0)
        return sum_enemy_point

    def render(self, mode='human', close=False):
        return 0
