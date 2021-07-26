import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.models import Sequential, load_model
env = gym.make('SpaceInvaders-v0')

env = gym.wrappers.FrameStack(env, 4)
print(env.observation_space)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, (8,8), strides=(4,4), activation='relu')
    self.conv2 = Conv2D(64, (4,4), strides=(2,2), activation='relu')
    self.conv3 = Conv2D(64, (3,3), activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(512, activation='relu')
    self.d2 = Dense(256, activation='relu')
    self.d3 = Dense(6, activation='linear')

  def call(self, x):
    x = tf.cast(x, tf.float32)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

class dqnAgent():
    def __init__(self, env):
        self.eps = 1.0
        self.env = env
    def get_action(self, state, env, p_model):
        action_q = p_model(state)
        action_q = tf.math.reduce_sum(action_q, axis=0).numpy()
        action = np.argmax(action_q)
        if(random.random() < self.eps):
            return env.action_space.sample()
        else:
            return action
    def train(self, ep_len, p_model, t_model, experience):
        state, action, next_state, reward, done = experience[0], experience[1], experience[2], experience[3], experience[4]
        optimizer = keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            qs = p_model(state)
            qsa = qs[0,action]
            nqs = t_model(next_state)
            nqsa = tf.reduce_max(nqs)
            r = tf.constant(reward)
            discount = tf.constant(0.99)
            tqsa = tf.add(r,tf.multiply(discount,nqsa))
            loss = tf.square(tf.subtract(tqsa,qsa))
        gradients = tape.gradient(loss, p_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, p_model.trainable_variables))
        if(ep_len%10 == 0):
            t_model = p_model
        if(done==True):
            self.eps = self.eps*0.95

Agent = dqnAgent(env)

state = env.reset()
p_model = MyModel()
t_model = p_model

t_reward = 0
info = 0
aggr_rewards = {'per_ep': []}
for episode in range(1000):
    ep_len = 0
    state = env.reset()
    done = False
    score = 0
    while not done:
        ep_len += 1
        action = Agent.get_action(state,env,p_model)
        next_state, reward, done, info = env.step([action])
        experience = (state, action, next_state, reward, done)
        Agent.train(ep_len, p_model, t_model, experience)
        state = next_state
        score += reward
        env.render()
    aggr_rewards['per_ep'].append(score)
    print(score)
env.close()
plt.plot(aggr_rewards['per_ep'])

import matplotlib.pyplot as plt
plt.plot(aggr_rewards['per_ep'])
