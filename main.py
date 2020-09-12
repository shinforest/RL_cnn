"""
Dependencies:
tensorflow r1.2
keras 2.2.4
"""
import os
import shutil
import random
import numpy as np
import tensorflow as tf
from collections import deque
from image_env import IMAGE
from DQN_Net import Policy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


grid=IMAGE()
policy=Policy(12,4)
t_action=grid.t_action
r_log=[]
for episode in range(1000):
    grid.resets(episode)
    state=grid.state
    done=grid.done
    actions=[]
    while not done:
        action=policy.choose_action(state)
        observation=grid.step(action,episode)
        n_state=observation[0]
        reward=observation[1]
        done=observation[2]
        policy.learn_act(state,reward,n_state)
        state=n_state
        r_log.append(reward)
        actions.append(t_action[action])
    print(episode,actions)
