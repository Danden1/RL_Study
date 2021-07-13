import math
import random
from collections import deque


import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

USE_CUDA = torch.cuda.is_available()

class PERBuffer(object):
    def __init__(self, capacity, prob_alpha = 0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,),dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state,0)
        next_state = np.expand_dims(next_state,0)


