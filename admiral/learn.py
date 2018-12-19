import numpy
import time

from admiral.env_interface import SeaGameJava

env = SeaGameJava('PythonRobo')
while True:
    env.step(numpy.array([1, 2]))
    time.sleep(0.005)
