import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor_Transformer(nn.Module):
    def __init(self, nb_states, nb_actions, init_w=3e-3):
        super(Actor_Transformer, self).__init__()
        # An actor that uses a transformer to encode the state
        # Dimension of Input: (num_contents * nb_states) + 1(update rate) 
        # Dimension of Output: (num_contents + 1) 
        
        # Input: [state]_1, [state]_2, ..., [state]_num_contents, [update rate]_1, [update rate]_2, ..., [update rate]_num_contents
        # Output: the probability of each content provider to update (0 represents no update and number > 0 represents the id of content provider to be updated)

        # The input will pass through two encoders of transformer
        # And cat(transfomer_output, (num_contents * 1(update rate))) to get the final output

        self.encoder1 = nn.TransformerEncoderLayer(d_model=nb_states, nhead=1, dim_feedforward=512)
        self.encoder2 = nn.TransformerEncoderLayer(d_model=nb_states, nhead=1, dim_feedforward=128)
        self.fc = nn.Linear(nb_states + 1, nb_actions)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, update_rate):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.fc(torch.cat([out, update_rate], 1))
        out = F.softmax(out, dim=1)
        return out

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
        # softmax
        out = F.softmax(out, dim=1)
        # out_tanh = out
        # tanh: [-1:1] -> [1:50]
        # out = (out + 1) * 0.5 * 49 + 1

        # print(f'Out_First: {out_first}, Out_Second: {out_second}, Out_Tanh: {out_tanh}, Out: {out}')

        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=1024, hidden2=512, hidden3=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2) # nn.Linear(hidden1+nb_actions, hidden2)
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