import socket
from time import sleep
import numpy as np

HOST = 'localhost'
PORT = 10000
name = 'tester'

BUFFER_SIZE = 4096
STAT = 'stat\n'.encode()
LEFT = 'left\n'.encode()
RIGHT = 'right\n'.encode()
UP = 'up\n'.encode()
DOWN = 'down\n'.encode()
RESET = 'reset\n'.encode()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data = 'login '+name + '\n'
s.sendall(data.encode())

while True:
    s.sendall(STAT)
    recv = s.recv(BUFFER_SIZE)
    if 'finished' in recv.decode():
        s.sendall(RESET)
    s.sendall(UP)
    s.sendall(UP)
