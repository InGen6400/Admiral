from typing import List, Dict, Any

import gym
import gym.spaces
import numpy as np
import socket

from admiral.const import LOGIN, STAT, RESET, NONE, UP, RIGHT, DOWN, LEFT
from numpy.core.multiarray import ndarray
from admiral.ship_agent import ShipAgent

HOST = 'localhost'
PORT = 10000
BUFFER_SIZE = 8192

NUM_NO_AI = 5
NAME_NO_AI = '*'

MAX_POINT = 1200
MAX_TANK = 4

RANK_REWARD = [100, 40, 20, -20, -40, -100]


def agent_name(num):
    return NAME_NO_AI + str(num)


class SeaGameJava(gym.core.Env):
    result_score: List[int]
    result_name: List[str]
    name: str
    tank_map: ndarray
    ship_map: ndarray

    def __init__(self, name):
        # 4方向+停止が2回分のアクション
        self.action_space = gym.spaces.Discrete(5 * 5)

        # 自分のpoint, 各座標の相手のポイント, 各座標のタンクのポイント
        self.observation_space = gym.spaces.Box(low=0, high=1200, shape=[256, 256, 2], dtype=np.uint16)

        # ソケット確保
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))

        # ログインする
        self.name = name

        self.agents = {}
        # 学習エージェント以外のログイン
        for n in range(0, NUM_NO_AI):
            self.sock.sendall(LOGIN.format(NAME_NO_AI + str(n)).encode())
            self.agents[agent_name(n)] = ShipAgent()

        self.ship_map = np.zeros((256, 256))
        self.tank_map = np.zeros((256, 256))
        self.my_x = 0
        self.my_y = 0
        self.point = 0
        self.capture = 0
        self.finished = False
        self.is_logged = False
        self.result_name = []
        self.result_score = []

    def step(self, action: int):
        if not self.is_logged:
            self.sock.sendall(LOGIN.format(self.name).encode())
            self.is_logged = True
        self.sock.sendall(ACTION_MEANING[action][0].format(self.name).encode())
        self.sock.sendall(ACTION_MEANING[action][1].format(self.name).encode())

        for a_name, agent in self.agents.items():
            self.sock.sendall(agent.move[0].format(a_name).encode())
            self.sock.sendall(agent.move[1].format(a_name).encode())

        self.sock.sendall(STAT.encode())

        sum_enemy_capture = self.reload()

        for agent in self.agents.values():
            agent.decide_move(self.ship_map, self.tank_map)

        # 終了していたら
        if self.finished:
            # ランキングに応じた報酬
            reward = RANK_REWARD[self.result_name.index(self.name)]
        else:
            # 試合中なら獲得ポイントを報酬に
            # 敵の獲得ポイントは平均値をマイナス報酬とする(重み軽め)
            reward = self.capture - sum_enemy_capture / NUM_NO_AI * 0.6

        return self.observe(), reward, self.finished, {}

    def observe(self):
        # 自分を中心にする
        ship_map = np.roll(self.ship_map, 128 - self.my_x, axis=1)
        ship_map = np.roll(ship_map, 128 - self.my_y, axis=0)
        tank_map = np.roll(self.tank_map, 128 - self.my_x, axis=1)
        tank_map = np.roll(tank_map, 128 - self.my_y, axis=0)
        self.my_x = 128
        self.my_y = 128
        # 自分は無視
        ship_map[128][128] = 0
        observation = np.dstack([ship_map, tank_map*MAX_POINT/MAX_TANK])
        return observation

    def reset(self):

        self.sock.sendall(RESET.encode())

        self.my_x = 0
        self.my_y = 0
        self.point = 0
        self.finished = False

        self.sock.sendall(STAT.encode())
        self.reload()

        obs = self.observe()
        return obs

    # サーバーから情報を取得して状態を更新する
    def reload(self):
        recv: bytes = self.sock.recv(BUFFER_SIZE)
        lines = recv.decode().split('\n')
        line_num = 0

        # 読み飛ばし
        while "ship_info" not in lines[line_num]:
            if "finished" in lines[line_num]:
                self.result_name.clear()
                self.result_score.clear()
                line_num = line_num + 1
                while "." not in lines[line_num]:
                    args: List[str] = lines[line_num].split(' ')
                    name = args[0]
                    point = int(args[1])
                    self.result_name.append(name)
                    self.result_score.append(point)
                    line_num = line_num + 1
                self.finished = True
                return 0
            line_num = line_num + 1

        line_num = line_num + 1
        self.ship_map.fill(0)
        sum_enemy_capture = 0
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
                self.capture = point - self.point
                self.point = point
            # 敵エージェント
            else:
                self.agents[name].x = x
                self.agents[name].y = y
                self.agents[name].capture = point - self.agents[name].point
                self.agents[name].point = point
                sum_enemy_capture = sum_enemy_capture + self.agents[name].capture

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

        return sum_enemy_capture

    def render(self, mode='human', close=False):
        return 0


ACTION_MEANING = {
    0: [NONE, NONE],
    1: [UP, NONE],
    2: [RIGHT, NONE],
    3: [DOWN, NONE],
    4: [LEFT, NONE],
    5: [NONE, UP],
    6: [UP, UP],
    7: [RIGHT, UP],
    8: [DOWN, UP],
    9: [LEFT, UP],
    10: [NONE, RIGHT],
    11: [UP, RIGHT],
    12: [RIGHT, RIGHT],
    13: [DOWN, RIGHT],
    14: [LEFT, RIGHT],
    15: [NONE, DOWN],
    16: [UP, DOWN],
    17: [RIGHT, DOWN],
    18: [DOWN, DOWN],
    19: [LEFT, DOWN],
    20: [NONE, LEFT],
    21: [UP, LEFT],
    22: [RIGHT, LEFT],
    23: [DOWN, LEFT],
    24: [LEFT, LEFT],
}
