import random
from typing import List, Tuple

import numpy as np

from admiral.const import LEFT, UP, DOWN, NONE, RIGHT, Commands


def tank2_weighted_tank(elem):
    return (128 - abs(elem[0] - 128)) + (128 - abs(elem[1] - 128))

DIST = [[0 for i in range(256)] for j in range(256)]
for j in range(0, 256):
    for i in range(0, 256):
        DIST[j][i] = tank2_weighted_tank([j, i])
DIST = np.array(DIST)

DIST_X = []
for i in range(0, 256):
    DIST_X[i] = 128 - abs(i - 128)

MODE_WEIGHTED_NEAR = 0  # スコア重み付け距離
MODE_NEAR = 1  # 距離
MODE_NEAR_BIGGEST = 2  # スコアの高いもの優先で近いもの
MODE_RANDOM = 3  # ランダム
MODE_ESCAPE_4DIR = 4  # 他の船から離れつつスコア重み付け距離
MODE_WEIGHTED_4DIR = 5  # 4方向に関して重み付けを合計して移動


class ShipAgent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.point = 0
        self.capture = 0
        self.move = [NONE] * 2
        self.mode = MODE_WEIGHTED_NEAR

    def decide_move(self, ship_map, tank_map: np.ndarray):
        # 自分中心に回転
        # ship_map = np.roll(ship_map, 128-self.x, axis=1)
        # ship_map = np.roll(ship_map, 128-self.y, axis=0)
        # tank_map = np.roll(tank_map, 128-self.x, axis=1)
        # tank_map = np.roll(tank_map, 128-self.y, axis=0)

        # 10%の確率でランダム移動
        if random.random() < 0.1:
            self.move[0] = Commands[random.randint(0, 4)]
            self.move[1] = Commands[random.randint(0, 4)]
        else:
            self.move[0] = NONE
            self.move[1] = NONE
            if self.mode == 0:
                self.move = self.decide_weighted_near(tank_map)
            elif self.mode == 1:

    def decide_weighted_near(self, tank_map):
        best_x = -1
        best_y = -1
        best_tank = 1000000
        y_index, x_index = np.where(tank_map != 0)
        for y, x in zip(y_index, x_index):
            tank = DIST[y - self.y][x - self.x] * 12 / tank_map[y][x]
            if tank < best_tank:
                best_tank = tank
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NONE, NONE]
        return self.target_to_dir([best_x, best_y])


    def decide_biggest_near(self, tank_map):
        best_x = -1
        best_y = -1
        best_dist = 10000
        best_tank = -100
        y_index, x_index = np.where(tank_map != 0)
        for y, x in zip(y_index, x_index):
            tank = tank_map[y][x]
            dist = DIST[y - self.y][x - self.x]
            if tank >= best_tank and dist < best_dist:
                best_tank = tank
                best_dist = dist
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NONE, NONE]
        return self.target_to_dir([best_x, best_y])

    def decide_near(self, tank_map):
        best_x = -1
        best_y = -1
        best_tank = 1000000
        y_index, x_index = np.where(tank_map != 0)
        for y, x in zip(y_index, x_index):
            tank = DIST[y - self.y][x - self.x]
            if tank < best_tank:
                best_tank = tank
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NONE, NONE]
        return self.target_to_dir([best_x, best_y])

    @staticmethod
    def decide_random():
        return [Commands[random.randint(0,4)], Commands[random.randint(0,4)]]

    def decide_escape(self, ship_map, tank_map):
        best_x = -1
        best_y = -1
        dir_score = []
        y_index, x_index = np.where(ship_map != 0)
        for y, x in zip(y_index, x_index):
            dx = DIST_X[x - self.x]
            dy = DIST_X[y - self.y]
            if dx + dy < best_dist:
                best_tank = tank
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NONE, NONE]
        return self.target_to_dir([best_x, best_y])

    def target_to_dir(self, target_pos: List[float]) -> List[str]:
        ret = [NONE, NONE]
        dx = target_pos[1] - self.x
        dy = target_pos[0] - self.y
        # X移動のほうが遠い
        if abs(dx) > abs(dy):
            if dx < 0:
                ret[0] = LEFT
                dx = dx + 10
            else:
                ret[0] = RIGHT
                dx = dx - 10
        else:
            if dy < 0:
                ret[0] = DOWN
                dy = dy + 10
            else:
                ret[0] = UP
                dy = dy - 10

        # 二回目の移動
        if abs(dx) < 10 and abs(dy) < 10:
            ret[1] = NONE
            if abs(dx) > abs(dy):
                if dx < 0:
                    ret[1] = LEFT
                else:
                    ret[1] = RIGHT
        return ret

    @staticmethod
    def get_quadrant(x, y):
        if x >= 0:
            if y >= 0:
                if y < x:
                    return 0
                else:
                    return 3
            else:
                if -y < x:
                    return 1
                else:
                    return 0
        else:
            if y >= 0:
                # x:- y:+
                if y < -x:
                    return 2
                else:
                    return 3
            else:
                # x:- y:-
                if y < x:
                    return 1
                else:
                    return 2