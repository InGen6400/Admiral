import numpy
import time

from admiral.env_interface import SeaGameJava

env = SeaGameJava('PythonRobo')
while True:
    obs, rew, done = env.step(numpy.array([1, 2]))
    if done:
        env.reset()
