import socket
from time import sleep

HOST = 'localhost'
PORT = 10000
name = 'tester'

BUFFER_SIZE = 4096
STAT = 'stat\n'.encode()
LEFT = 'left\n'.encode()
RIGHT = 'right\n'.encode()
UP = 'up\n'.encode()
DOWN = 'down\n'.encode()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data = 'login '+name + '\n'
s.sendall(data.encode())

while True:
    s.sendall(STAT)
    s.sendall(UP)
    s.sendall(UP)
    sleep(0.5)

