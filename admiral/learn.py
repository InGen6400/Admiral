import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Reshape, SeparableConv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


from admiral.env_interface import SeaGameJava

env = SeaGameJava('PythonRobo')

print(env.observation_space.shape)

nb_actions = env.action_space.n


model = Sequential()
model.add(Reshape((256, 256, 2), input_shape=(1,) + env.observation_space.shape))
model.add(SeparableConv2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=6000, window_length=1)

policy = EpsGreedyQPolicy(eps=0.1)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1200,
               target_model_update=1e-2, policy=policy, train_interval=100)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=6000, action_repetition=3, visualize=False, verbose=1, nb_max_episode_steps=600)
dqn.test(env, nb_episodes=10, visualize=False)
