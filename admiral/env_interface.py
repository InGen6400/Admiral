from typing import List

import gym
import gym.spaces
import numpy as np
import socket

from admiral.const import LOGIN, LEFT, RIGHT, UP, DOWN, Commands, STAT, RESET
from numpy.core.multiarray import ndarray
from admiral.ship_agent import ShipAgent

HOST = 'localhost'
PORT = 10000
BUFFER_SIZE = 8192

NUM_NO_AI = 5
NAME_NO_AI = '*'


def agent_name(num):
    return NAME_NO_AI + str(num)


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
        self.sock.sendall(LOGIN.format(name).encode())

        self.agents = {}
        # 学習エージェント以外のログイン
        for n in range(0, NUM_NO_AI):
            self.sock.sendall(LOGIN.format(NAME_NO_AI+str(n)).encode())
            self.agents[agent_name(n)] = ShipAgent()

        self.ship_map = np.zeros((256, 256))
        self.tank_map = np.zeros((256, 256))
        self.my_x = 0
        self.my_y = 0
        self.my_point = 0
        self.finished = False

    def step(self, action: np.ndarray):
        self.sock.sendall(Commands[action[0]].format(self.name).encode())
        self.sock.sendall(Commands[action[1]].format(self.name).encode())

        for a_name, agent in self.agents.items():
            self.sock.sendall(agent.move[0].format(a_name).encode())
            self.sock.sendall(agent.move[1].format(a_name).encode())

        self.sock.sendall(STAT.encode())

        sum_enemy_point = self.reload()

        for agent in self.agents.values():
            agent.decide_move(self.ship_map, self.tank_map)

        return self.observe(), self.my_point - sum_enemy_point, self.finished

    def observe(self):
        # 自分を中心にする
        ship_map = np.roll(self.ship_map, 128-self.my_x, axis=1)
        ship_map = np.roll(ship_map, 128-self.my_y, axis=0)
        tank_map = np.roll(self.tank_map, 128-self.my_x, axis=1)
        tank_map = np.roll(tank_map, 128-self.my_y, axis=0)
        self.my_x = 128
        self.my_y = 128
        # 自分は無視
        ship_map[128][128] = 0
        observation = {
            'myPoint': self.my_point,
            'eShip': ship_map,
            'tank': tank_map
        }
        return observation

    def reset(self):

        self.sock.sendall(RESET.encode())

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
        while "ship_info" not in lines[line_num]:
            if "finished" in lines[line_num]:
                self.finished = True
                return 0
            line_num = line_num + 1

        line_num = line_num + 1
        self.ship_map.fill(0)
        sum_enemy_point = 0
        # 敵の船情報の取得
        while "." not in lines[line_num]:
            args: List[str] = lines[line_num].split(' ')
            name = args[0]
            x = int(args[1])
            y = int(args[2])
            point = int(args[3])

            self.ship_map[y][x] = point
            # 自分の情報を保存
            if name == self.name:
                self.my_x = x
                self.my_y = y
                self.my_point = point
            # 敵エージェント
            else:
                self.agents[name].x = x
                self.agents[name].y = y
                sum_enemy_point = sum_enemy_point + point

            line_num = line_num + 1

        # 読み飛ばし
        while "energy_info" not in lines[line_num]:
            line_num = line_num + 1

        line_num = line_num + 1
        self.tank_map.fill(0)
        # タンク情報の取得
        while "." not in lines[line_num]:
            args: List[str] = lines[line_num].split(' ')
            x = int(args[0])
            y = int(args[1])
            point = int(args[2])

            self.tank_map[y][x] = point
            line_num = line_num + 1

        return sum_enemy_point

    def render(self, mode='human', close=False):
        return 0
