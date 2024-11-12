import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)
# Actor: Two hidden laysers with 512 and 128.
# ReLU are used as the activation functions for the hidden layers. The output layer uses tanh and scaling to ensure the threshold is within [1, max_age]
# Critic: Three hidden layers with 1024, 512, and 300. ReLU are used as the activation functions for the hidden layers.
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=512, hidden2=128, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out_first = out
        out = self.fc2(out)
        out = self.relu(out)
        out_second = out
        out = self.fc3(out)
        out = self.tanh(out)
        out_tanh = out
        # tanh: [-1:1] -> [1:50]
        out = (out + 1) * 0.5 * 49 + 1

        # print(f'Out_First: {out_first}, Out_Second: {out_second}, Out_Tanh: {out_tanh}, Out: {out}')

        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=1024, hidden2=512, hidden3=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+1, hidden2) # nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out