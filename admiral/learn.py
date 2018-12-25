import os

import numpy as np
import time
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, concatenate, Input, Permute, MaxPooling2D, \
    AveragePooling2D, Reshape
from keras.optimizers import Adam
from keras.utils import plot_model

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor

from admiral.env_interface import SeaGameJava


WEIGHT_FILE = 'weight.h5f'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)

env = SeaGameJava('PythonRobo')

nb_actions = env.action_space.n

shape = (256, 256, 1)
model_ship = Sequential()
model_ship.add(Reshape(shape, input_shape=(1,) + shape))
model_ship.add(AveragePooling2D((4, 4), (2, 2)))
model_ship.add(Flatten())
model_ship.add(Dense(16))
model_ship.add(Activation('relu'))
model_ship.add(Dense(16))
model_ship.add(Activation('relu'))
model_ship_input = Input(shape=(1,)+shape, name='ship_map')
model_ship_encoded = model_ship(model_ship_input)

plot_model(model_ship, to_file='model_ship.png')

model_tank = Sequential()
model_tank.add(Reshape(shape, input_shape=(1,) + shape))
model_tank.add(AveragePooling2D((4, 4), (2, 2)))
model_tank.add(Flatten())
model_tank.add(Dense(16))
model_tank.add(Activation('relu'))
model_tank.add(Dense(16))
model_tank.add(Activation('relu'))
model_tank_input = Input(shape=(1,)+shape, name='tank_map')
model_tank_encoded = model_tank(model_tank_input)

plot_model(model_tank, to_file='model_tank.png')

con = concatenate([model_ship_encoded, model_tank_encoded])

hidden = Dense(16, activation='relu')(con)
hidden = Dense(16, activation='relu')(hidden)

output = Dense(nb_actions, activation='linear')(hidden)

model_final = Model(inputs=[model_ship_input, model_tank_input], outputs=output)
# print(model.summary())
plot_model(model_final, to_file='model.png')

memory = SequentialMemory(limit=600*10, window_length=1)

policy = EpsGreedyQPolicy(eps=0.01)

dqn = DQNAgent(model=model_final, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1200,
               target_model_update=1e-2, policy=policy, train_interval=600, batch_size=512)
if os.path.exists(WEIGHT_FILE):
    dqn.load_weights(WEIGHT_FILE)
dqn.processor = MultiInputProcessor(2)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=600000, visualize=False, verbose=2, nb_max_episode_steps=None)

dqn.save_weights(WEIGHT_FILE, True)
dqn.test(env, nb_episodes=10, visualize=True)
