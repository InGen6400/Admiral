import socket
from time import sleep
import numpy as np

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
