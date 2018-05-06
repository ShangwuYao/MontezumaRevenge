#!/usr/bin/env python
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop, Adam
import numpy as np
import random
from collections import deque
import gym
import pickle
import os, sys, copy, argparse
import cv2
from keras import backend as K
from keras.layers import Lambda
from keras.models import Model

import pdb



class RLAgents():
	"""
	Different reinforcement learning for different mini-tasks
	dynamically adding new agents
	"""
	def __init__(self, state_dim, action_cnt, env_name='MZ'):
		"""
		state_dim: dimension of states, tuple: (84, 84, 4)
		action_cnt: dimension of actions, scala
		env_name: environment name, for keeping track
		"""
		self.state_dim = state_dim
		self.action_cnt = action_cnt
		self.env_name = env_name
		self.Agent_Dict = {}

	def _get_agent(self, mini_task):
		"""
		mini_task: string for mini_task, each agent correspond to a mini_task
		"""
		if mini_task not in self.Agent_Dict:
			self.Agent_Dict[mini_task] = Imitation_Agent(self.state_dim, self.action_cnt, self.env_name, mini_task, 
												   lr=1e-3, train=True, save_every_step=1000, debug=False)
		return self.Agent_Dict[mini_task]

	def feedback(self, mini_task, feedbacks):
		agent = self._get_agent(mini_task)
		agent.step(feedbacks)

	def execute(self, mini_task, states):
		agent = self._get_agent(mini_task)

		action = agent.predict(states)
		
		return action


class Imitation_Agent():
	"""
	Policy network imitation learning agent.
	"""
	def __init__(self, state_dim, action_cnt, env_name, mini_task, lr=1e-3,
				 train=True, save_every_step=1000, debug=False):
		"""
		Initialize a imitation learning agent.

		@param state_dim The dimension of states. (84, 84, 4)
		@param action_cnt The number of actions. scala
		@param env_name The name of the environment.
		@param mini_task The mini task for the network.
		@param train The switch to train the model. (default: True)
		"""

		self.state_dim = state_dim
		self.action_cnt = action_cnt
		# initialize the full actions array
		self.all_actions = [i for i in range(self.action_cnt)]

		self.env_name = env_name
		self.mini_task = mini_task

		self.debug = debug
		self.should_train = train

		# initialize the policy network and load weights if exist
		self.batch_size = 32
		self.learning_rate = lr
		self.network = PolicyNetwork(self.state_dim, self.action_cnt, 
														self.env_name, self.mini_task,
														self.batch_size, self.learning_rate)
		self.step_cnt = 0
		self.save_every_step = save_every_step

		# load the saved model weights
		if os.path.isfile(self.network.default_weights_path):
			self.network.load_weights()

	
	def predict(self, s):
		"""
		Predict the policy for the input state.

		@param s The current state.
		@return The action to take learned from expert.
		"""
		action_logits = self.network.predict(s)
		action = np.argmax(action_logits[0]) # TODO
		return action
	
	def step(self, feedbacks):
		"""
		Step forward to the next state.

		feedbacks: a tuple. 
					example: 
						(state_images, actions, next_states, rewards, done)
						state_images and next_states should be of shape (1, 84, 84, 4)
						action should be a scala
						rewards should be a scala, the sum of rewards for the 4 frames
						done should be a boolean
		"""
		state_images, actions, _, _, _ = feedbacks

		action_idx = actions[0] # all actions the same
		action = np.zeros((1, self.action_cnt))
		action[0, action_idx] = 1

		self.network.train(state_images, action)
		
		# save network
		self.step_cnt += 1
		if self.should_train and self.step_cnt % self.save_every_step == 0 and self.step_cnt != 0:
			# save the latest model weights
			self.network.save_weights()



class PolicyNetwork():
	def __init__(self, state_dim, action_cnt, env_name, suffix, batch_size=32, learning_rate=1e-4, debug=False):
		"""
		Initialize a policy network instance.

		@param state_dim The dimension of state space. (84, 84, 4)
		@param action_cnt The dimension of action space. scala
		@param env_name The name of the environment.
		@param suffix The test case suffix for the network.
		@param batch_size The size of the mini-batch in mini-batch gradient descent 
											optimization for training the network. (default: 32)
		@param learning_rate The learning rate for training the network. 
												 (default: 1e-4)
		"""

		self.state_dim = state_dim
		self.action_cnt = action_cnt

		self.env_name = env_name
		self.suffix = suffix

		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.debug = debug

		# construct default file paths
		self.default_model_path = 'save_PN_' + self.env_name + '_' + self.suffix + '_model.h5'
		self.default_weights_path = 'save_PN_' + self.env_name + '_' + self.suffix + '_weights.h5'

		# build the keras model
		self.model = self._build_model()

	def _build_model(self):
		"""
		(Internal)
		Build the keras model of the policy network.

		@return A built keras model.
		"""

		model_input = Input(shape=self.state_dim)

		x = Conv2D(filters=32, kernel_size=8, strides=4, padding="valid", activation="relu")(model_input)
		x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid", activation="relu")(x)
		x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu")(x)
		x = Flatten()(x)
		x = Dense(activation='relu', units=512)(x)
		action_logits = Dense(activation='softmax', units=self.action_cnt)(x)

		model = Model(input=model_input, output=action_logits)

		optimizerRMSprop = RMSprop(lr=self.learning_rate)
		optimizerAdam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)
		optimizer = optimizerAdam # TODO

		model.compile(loss='categorical_crossentropy', optimizer=optimizer)

		return model

	def train(self, X, Y, epochs=1, verbose=0):
		self.model.fit(X, Y, epochs=epochs, verbose=verbose)

	def predict(self, state):

		prediction = self.model.predict(state)
		return prediction

	def save_model(self, file_path=None):
		"""
		Save the keras model as a file.

		@param file_path The file path to save the keras model. (optional)
		@return The path to the saved keras model file.
		"""
		
		if file_path is None:
			file_path = self.default_model_path

		if self.debug:
			print("-----saving model to {}".format(file_path))
			
		self.model.save(file_path)

		return file_path

	def save_weights(self, file_path=None):
		"""
		Save the keras model weights as a file.

		@param file_path The file path to save the keras model weights. (optional)
		@return The path to the saved keras weights file.
		"""

		if file_path is None:
			file_path = self.default_weights_path

		if self.debug:
			print("-----saving weights to {}".format(file_path))
			
		self.model.save_weights(file_path)

		return file_path

	def load_model(self, file_path=None):
		"""
		Load an existing keras model from a file.

		@param file_path The path to an existing keras model file. (optional)
		"""

		if file_path is not None:
			if self.debug:
				print("-----loading model from {}".format(file_path))
			self.model.load(file_path)
		else:
			if self.debug:
				print("-----loading model from {}".format(self.default_model_path))
			self.model.load(self.default_model_path)

	def load_weights(self, file_path=None):
		"""
		Load existing keras model weights from a file.

		@param file_path The path to an existing keras weights file. (optional)
		"""

		if file_path is not None:
			if self.debug:
				print("-----loading weights from {}".format(file_path))
			self.model.load_weights(file_path)
		else:
			if self.debug:
				print("-----loading weights from {}".format(self.default_weights_path))
			self.model.load_weights(self.default_weights_path)


def main():
	env_name = 'SpaceInvaders-v0'
	env = gym.make(env_name)

	state_dim = (84, 84, 4)
	action_cnt = env.action_space.n

	rlagents = RLAgents(state_dim, action_cnt, '_debug_')

	mini_task = "say hi"
	states = np.random.rand(1, 84, 84, 4)
	feedbacks = (states, [5, 4, 3, 2], states, 100, True)

	rlagents.feedback(mini_task, feedbacks)

	action = rlagents.execute(mini_task, states)
	print("------action: {}------".format(action))


if __name__ == '__main__':
	main()
