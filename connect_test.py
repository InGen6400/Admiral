import socket
from time import sleep

HOST = 'localhost'
PORT = 10000
name = 'tester'

BUFFER_SIZE = 4096
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data = 'login '+name
s.sendall(data.encode())

s.sendall('stat'.encode())
s.sendall(UP.encode())

sleep(3)

s.sendall('stat'.encode())
s.sendall(UP.encode())

sleep(0.5)

