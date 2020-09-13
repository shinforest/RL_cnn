"""
Dependencies:
tensorflow r1.2
keras 2.2.4
"""
import os
import shutil
import random
import numpy as np
# import tensorflow as tf
from collections import deque
from image_env import IMAGE
from DQN_Net import Policy
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from load_model import Model
m = Model()
model = m.load_model()


grid=IMAGE()
policy=Policy(7500,7)
t_action=grid.t_action
r_log=[]
for episode in range(1000):
    grid.resets(episode)
    state=grid.state
    done=grid.done
    actions=[]
    while not done:
        action=policy.choose_action(state)
        observation=grid.step(action,episode,model)
        n_state=observation[0].reshape(-1)
        reward=observation[1]
        done=observation[2]
        print(reward)
        print(state.shape)
        print(n_state.shape)
        policy.learn_act(state,reward,n_state)
        state=n_state
        r_log.append(reward)
        actions.append(t_action[action])
    print(episode,actions)

np.save("r_log", r_log)
