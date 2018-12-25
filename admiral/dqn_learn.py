import os
import gym
import time
import random
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from collections import deque
from operator import itemgetter

from keras import Input, Model
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, AveragePooling2D, Flatten, Activation, Concatenate, Reshape
import matplotlib.pyplot as plt

from admiral.env_interface import SeaGameJava

plt.style.use('seaborn')
plt.rcParams['font.family'] = 'IPAexGothic'


class DQN(object):

    def __init__(self, env, agent_hist_len=4, memory_size=2000,
                 replay_start_size=32, gamma=0.99, eps=1.0, eps_min=1e-4,
                 final_expl_step=1000, mb_size=32, C=100, n_episodes=400,
                 max_steps=500):

        self.env = env
        self.path = './data/' + str(env)
        self.agent_hist_len = agent_hist_len
        self.memory_size = memory_size
        self.replay_start_size = replay_start_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.final_expl_step = final_expl_step
        self.eps_decay = (eps - eps_min) / final_expl_step
        self.mb_size = mb_size
        self.C = C
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self._init_memory()

    @staticmethod
    def _flatten_deque(d):
        return np.array(list(chain(*d)))

    @staticmethod
    def _get_optimal_action(network, agent_state_hist):
        return np.argmax(network.predict(agent_state_hist)[0])

    def _get_action(self, agent_state_hist=None):
        if agent_state_hist is None:
            return self.env.action_space.sample()
        else:
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
            if np.random.random() < self.eps:
                return self.env.action_space.sample()
            else:
                return self._get_optimal_action(self.Q, agent_state_hist)

    def _remember(self, agent_state_hist, action, reward, new_state, done):
        self.memory.append([agent_state_hist, action, reward, new_state if not done else None])

    def _init_memory(self):
        print('Initializing replay memory: ', end='')
        self.memory = deque(maxlen=self.memory_size)
        while True:
            state = self.env.reset()
            agent_state_hist = deque(maxlen=self.agent_hist_len)
            agent_state_hist.append(state)
            while True:
                action = self._get_action(agent_state_hist=None)
                new_state, reward, done, _ = self.env.step(action)
                if len(agent_state_hist) == self.agent_hist_len:
                    self._remember(agent_state_hist, action, reward, new_state, done)
                if len(self.memory) == self.replay_start_size:
                    print('done')
                    return
                if done:
                    break
                state = new_state
                agent_state_hist.append(state)

    def _build_network(self):
        ship_input = Input(shape=(256, 256, 1))
        ship_model = Sequential()
        ship_model.add(AveragePooling2D((4, 4), (2, 2)))
        ship_model.add(Flatten())
        ship_model.add(Dense(32))
        ship_model.add(Activation('relu'))
        ship_model.add(Dense(16))
        ship_model.add(Activation('relu'))
        ship_final = ship_model(ship_input)

        plot_model(ship_model, to_file='model_ship.png', show_shapes=True)

        tank_input = Input(shape=(256, 256, 1))
        tank_model = Sequential()
        tank_model.add(AveragePooling2D((4, 4), (2, 2)))
        tank_model.add(Flatten())
        tank_model.add(Dense(32))
        tank_model.add(Activation('relu'))
        tank_model.add(Dense(16))
        tank_model.add(Activation('relu'))
        tank_final = tank_model(tank_input)

        plot_model(tank_model, to_file='model_tank.png', show_shapes=True)

        con = Concatenate()([ship_final, tank_final])

        hidden = Dense(16, activation='relu')(con)
        hidden2 = Dense(16, activation='relu')(hidden)

        output = Dense(self.env.action_space.n, activation='linear')(hidden2)

        model = Model(inputs=[ship_input, tank_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    def _clone_network(self, nn):
        clone = self._build_network()
        clone.set_weights(nn.get_weights())
        return clone

    def _get_samples(self):
        samples = random.sample(self.memory, self.mb_size)
        agent_state_hists = np.array([s[0] for s in samples])
        Y = self.target_Q.predict(agent_state_hists)
        actions = [s[1] for s in samples]
        rewards = np.array([s[2] for s in samples])
        future_rewards = np.zeros(self.mb_size)
        new_states_idx = [i for i, s in enumerate(samples) if s[3] is not None]
        new_states = np.array([s[3] for s in itemgetter(*new_states_idx)(samples)])
        new_agent_hists = np.hstack(
            [agent_state_hists[new_states_idx, self.env.observation_space.shape[0]:],
             new_states])
        future_rewards[new_states_idx] = np.max(
            self.target_Q.predict(new_agent_hists), axis=1)
        rewards += self.gamma * future_rewards
        for i, r in enumerate(Y):
            Y[i, actions[i]] = rewards[i]
        return agent_state_hists, Y

    def _replay(self):
        agent_state_hists, Y = self._get_samples()
        for i in range(self.mb_size):
            self.Q.train_on_batch([agent_state_hists['ship_map'][i, :].reshape(1, -1),
                                   agent_state_hists['tank_map'][i, :].reshape(1, -1)],
                                  Y[i, :].reshape(1, -1))

    def learn(self, render=False, verbose=True):

        self.Q = self._build_network()
        self.target_Q = self._clone_network(self.Q)

        if verbose:
            print('Learning target network:')
        self.scores = []
        for episode in range(self.n_episodes):
            state = self.env.reset()
            agent_state_hist = deque(maxlen=self.agent_hist_len)
            agent_state_hist.append(state)
            score = 0
            for step in range(self.max_steps):
                if render:
                    self.env.render()
                if len(agent_state_hist) < self.agent_hist_len:
                    action = self._get_action(agent_state_hist=None)
                else:
                    action = self._get_action(agent_state_hist)
                new_state, reward, done, _ = self.env.step(action)
                if verbose:
                    print('episode: {:4} | step: {:3} | memory: {:6} | \
eps: {:.4f} | action: {} | reward: {: .1f} | best score: {: 6.1f} | \
mean score: {: 6.1f}'.format(
                        episode + 1, step + 1, len(self.memory), self.eps, action, reward,
                        max(self.scores) if len(self.scores) != 0 else np.nan,
                        np.mean(self.scores) if len(self.scores) != 0 else np.nan),
                        end='\r')
                score += reward
                if len(agent_state_hist) == self.agent_hist_len:
                    self._remember(agent_state_hist, action, reward, new_state, done)
                    self._replay()
                if step % self.C == 0:
                    self.target_Q = self._clone_network(self.Q)
                if done:
                    self.scores.append(score)
                    break
                state = new_state
                agent_state_hist.append(state)

        self.target_Q.save(self.path + '_model.h5')
        with open(self.path + '_scores.pkl', 'wb') as f:
            pickle.dump(self.scores, f)

    def plot_training_scores(self):
        with open(self.path + '_scores.pkl', 'rb') as f:
            scores = pd.Series(pickle.load(f))
        avg_scores = scores.cumsum() / (scores.index + 1)
        plt.figure(figsize=(12, 6))
        n_scores = len(scores)
        plt.plot(range(n_scores), scores, color='gray', linewidth=1)
        plt.plot(range(n_scores), avg_scores, label='平均')
        plt.legend()
        plt.xlabel('学習エピソード')
        plt.ylabel('スコア')
        plt.title(str(self.env))
        plt.margins(0.02)
        plt.tight_layout()
        plt.show()

    def run(self, render=True):

        fname = self.path + '_model.h5'
        if os.path.exists(fname):
            self.target_Q = load_model(fname)
        else:
            print('Q-network not found. Start learning.')
            self.learn()

        state = self.env.reset()
        agent_state_hist = deque(maxlen=self.agent_hist_len)
        agent_state_hist.extend([state] * self.agent_hist_len)
        score = 0
        while True:
            if render:
                self.env.render()
            action = self._get_optimal_action(self.target_Q, agent_state_hist)
            new_state, reward, done, _ = self.env.step(action)
            score += reward
            if done:
                print('{} score: {}'.format(str(self.env), score))
                return
            state = new_state
            agent_state_hist.append(state)
            time.sleep(0.05)


env = SeaGameJava('PythonRobot')
#env = gym.make('CartPole-v0')
dqn = DQN(env, n_episodes=100, max_steps=600)
dqn.learn(render=False, verbose=True)
dqn.plot_training_scores()

