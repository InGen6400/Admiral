import time

from admiral.env_interface import SeaGameJava

env = SeaGameJava('PythonRobo')
env.step([1, 0])
time.sleep(10)
env.step([2, 0])
