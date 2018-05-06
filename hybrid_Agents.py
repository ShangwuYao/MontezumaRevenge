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

from ImitationLearningAgents import Imitation_Agent
from RLAgents import DQN_Agent



class RLAgents():
	"""
	Different reinforcement learning for different mini-tasks
	dynamically adding new agents
	"""
	def __init__(self, state_dim, action_cnt, imitation_tasks, env_name='MZ'):
		"""
		state_dim: dimension of states, tuple: (84, 84, 4)
		action_cnt: dimension of actions, scala
		env_name: environment name, for keeping track
		"""
		self.state_dim = state_dim
		self.action_cnt = action_cnt
		self.env_name = env_name
		self.Agent_Dict = {}
		self.imitation_tasks = imitation_tasks

	def _get_agent(self, mini_task):
		"""
		mini_task: string for mini_task, each agent correspond to a mini_task
		"""
		if mini_task not in self.Agent_Dict:
			if mini_task in self.imitation_tasks:
				print("------new imitation learning task------")
				self.Agent_Dict[mini_task] = Imitation_Agent(self.state_dim, self.action_cnt, self.env_name, mini_task, 
												   lr=1e-3, train=True, save_every_step=1000, debug=False)
			else:
				print("------new DQN task------")
				self.Agent_Dict[mini_task] = DQN_Agent(self.state_dim, self.action_cnt, self.env_name, mini_task, 
												   epsilon_start=1., epsilon_end=0.1, 
												   epsilon_linear_red=0.0001, replay=True, 
												   gamma=0.99, train=True)
		return self.Agent_Dict[mini_task]

	def feedback(self, mini_task, feedbacks):
		agent = self._get_agent(mini_task)
		agent.step(feedbacks)

	def execute(self, mini_task, states):
		agent = self._get_agent(mini_task)

		if mini_task in self.imitation_tasks: # imitation learning
			action = agent.predict(states)
			return action

		else:                                 # DQN
			q_values = agent.predict(states)
			action = agent.epsilon_greedy_policy(q_values)
		
			return action


def main():
	env_name = 'SpaceInvaders-v0'
	env = gym.make(env_name)

	state_dim = (84, 84, 4)
	action_cnt = env.action_space.n

	imitation_tasks = set()
	imitation_tasks.add("say hi")
	rlagents = RLAgents(state_dim, action_cnt, imitation_tasks, '_debug_')

	mini_task = "say hi"
	states = np.random.rand(1, 84, 84, 4)
	feedbacks = (states, [5, 4, 3, 2], states, 100, True)

	rlagents.feedback(mini_task, feedbacks)

	action = rlagents.execute(mini_task, states)
	print("------action: {}------".format(action))


if __name__ == '__main__':
	main()
