import random
import numpy as np

from admiral.ship_agent import MASK_R, MASK_D, MASK_U, MASK_L


DIST_X = [-8] * 16
for i in range(0, 8):
    DIST_X[i] = i if i != 0 else -8
    DIST_X[-i] = -i
print(MASK_R[105+118:105+138, 12+118:12+138])
print(MASK_D[105+118:105+138, 12+118:12+138])
print(MASK_L[105+118:105+138, 12+118:12+138])
print(MASK_U[105+118:105+138, 12+118:12+138])
print(MASK_R[105+128][12+128])
print(MASK_D[105+128][12+128])
print(MASK_L[105+128][12+128])
print(MASK_U[105+128][12+128])
print(MASK_D[118:138, 118:138])


mask_default = np.zeros((16, 16))
y, x = np.ogrid[0:16, 0:16]
mask_r = (8 - np.abs(y - 8)) - (8 - (x - 8)) >= 0
MASK_R = mask_default.copy()
MASK_R[mask_r] = 1

mask_l = (8 - np.abs(y - 8)) - (8 + (x - 8)) > 0
MASK_L = mask_default.copy()
MASK_L[mask_l] = 1

MASK_U = np.ones((16, 16))
MASK_U[mask_r + mask_l] = 0
MASK_U[8:16, :] = 0

MASK_D = np.ones((16, 16))
MASK_D[mask_r + mask_l] = 0
MASK_D[0:8, :] = 0
'''
print(MASK_R)
print(MASK_D)
print(MASK_L)
print(MASK_U)
print(MASK_R[5][2])
print(MASK_D[5][2])
print(MASK_L[5][2])
print(MASK_U[5][2])
'''
'''
HOST = 'localhost'
PORT = 10000
name = 'tester'

BUFFER_SIZE = 4096
STAT = 'stat\n'
LEFT = 'left {}\n'
RIGHT = 'right {}\n'
UP = 'up {}\n'
DOWN = 'down {}\n'
RESET = 'reset\n'

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data = 'login '+name + '\n'
s.sendall(data.encode())
data = 'login '+'robot' + '\n'
s.sendall(data.encode())

while True:
    s.sendall(STAT.encode())
    recv = s.recv(BUFFER_SIZE)
    if 'finished' in recv.decode():
        s.sendall(RESET.encode())
    s.sendall(UP.format(name).encode())
    s.sendall(LEFT.format(name).encode())
    s.sendall(UP.format('robot').encode())
    s.sendall(RIGHT.format('robot').encode())
'''
'''
memory = deque(maxlen=4)
memory.append('123')
memory.append([12, 123])

print([t[0] for t in memory])
'''
'''

env = SeaGameJava('PythonRobo')
while True:
    obs, rew, done = env.step(numpy.array([1, 3]))
    if done:
        env.reset()
'''
'''
for _ in range(20):
    print(random.randint(0, 4))
'''



