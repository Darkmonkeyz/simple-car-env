import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random


# Define the Q-Network model 
class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped
##########################################################################################################################

#load the model
model = QNetwork(6, 9)
model.load_state_dict(torch.load("simple_driving_qlearning.pkl"))  # Load the trained model


state, info = env.reset()
#frames = []
#frames.append(env.render())

total_reward = 0

for i in range(400):  # Max steps
    with torch.no_grad():
        # Use the model to predict the best action
        q_values = model(torch.tensor(state, dtype=torch.float32))
        action = torch.argmax(q_values).item()

    # Take the action and observe the result
    state, reward, done, _, info = env.step(action)
    total_reward += reward
  
    if done:
        break

env.close()