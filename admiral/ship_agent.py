import numpy as np

from admiral.const import LEFT, UP, DOWN, NONE, RIGHT


class ShipAgent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.move = [NONE]*2

    def decide_move(self, ship_map, tank_map: np.ndarray):
        # 自分中心に回転
        #ship_map = np.roll(ship_map, 128-self.x, axis=1)
        #ship_map = np.roll(ship_map, 128-self.y, axis=0)
        tank_map = np.roll(tank_map, 128-self.x, axis=1)
        tank_map = np.roll(tank_map, 128-self.y, axis=0)
        self.x = 128
        self.y = 128

        best_x = -1
        best_y = -1
        best_tank = 1000000
        self.move[0] = NONE
        self.move[1] = NONE
        for y in range(0, 256):
            for x in range(0, 256):
                if tank_map[y][x] != 0:
                    tank = (abs(x-self.x) + abs(y-self.y)) * 12 / tank_map[y][x]
                    # 距離が小さいものを保存
                    if tank < best_tank:
                        best_tank = tank
                        best_x = x
                        best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return
        dx = best_x-self.x
        dy = best_y-self.y
        # X移動のほうが遠い
        if abs(dx) > abs(dy):
            if dx < 0:
                self.move[0] = LEFT
                dx = dx + 10
            else:
                self.move[0] = RIGHT
                dx = dx - 10
        else:
            if dy < 0:
                self.move[0] = DOWN
                dy = dy + 10
            else:
                self.move[0] = UP
                dy = dy - 10

        # 二回目の移動
        if abs(dx) < 10 and abs(dy) < 10:
            self.move[1] = NONE
            if abs(dx) > abs(dy):
                if dx < 0:
                    self.move[1] = LEFT
                else:
                    self.move[1] = RIGHT



