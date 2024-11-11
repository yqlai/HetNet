import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Actor, Critic
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from utils import *

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, config):
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Working on {self.device}')

        self.actor = Actor(self.nb_states-1, 1)
        self.actor_target = Actor(self.nb_states-1, 1)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.00001)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_target = Critic(self.nb_states, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.00001)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
 
        self.memory = SequentialMemory(limit=1000000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0, sigma=0.01)

        self.batch_size = 32
        self.tau = 0.01
        self.depsilon = 0 # 1.0 / 50000

        # ref_state is a reference state for the critic
        # make ref_state_batch by repeating ref_state
        self.ref_state_batch = np.array([config.config['ref_state'] for _ in range(self.batch_size)]).astype(np.float32)

        self.epsilon = 1.0
        self.s_t = None
        self.a_t = None
        self.is_training = True

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
    
    def update_policy(self):
        state_batch, action_batch, Lagrangian_batch, next_state_batch, _ = self.memory.sample_and_split(self.batch_size)
        
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True, device=self.device),
            self.actor_target(to_tensor(next_state_batch[:, 1:], volatile=True, device=self.device))
        ])
        next_q_values.volatile = False
        ref_values = self.critic_target([
            to_tensor(self.ref_state_batch, volatile=True, device=self.device),
            self.actor_target(to_tensor(self.ref_state_batch[:, 1:], volatile=True, device=self.device))
        ])
        ref_values.volatile = False

        target_q_batch = to_tensor(Lagrangian_batch, device=self.device) + next_q_values - ref_values

        self.critic.zero_grad()
        q_batch = self.critic([to_tensor(state_batch, device=self.device), to_tensor(action_batch, device=self.device)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss = self.critic([
            to_tensor(state_batch, device=self.device),
            self.actor(to_tensor(state_batch[:, 1:], device=self.device))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
    
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def observe(self, l_t, s_t1):
        if self.s_t is not None:
            self.memory.append(self.s_t, self.a_t, l_t, False)
        self.s_t = s_t1

    # Action
    def random_action(self):
        action = np.random.randint(self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        threshold = self.actor(to_tensor(np.array([s_t[1:]]).astype(np.float32))).squeeze(0)
        if s_t[0] > threshold:
            action = 1
        else:
            action = 0

        # action = self.actor(to_tensor(np.array([s_t[1:]]).astype(np.float32))).squeeze(0)
        # action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        # action = np.clip(action, 0, self.nb_actions)
        self.a_t = action

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        return action, threshold

    def reset(self, s_t):
        self.s_t = s_t
        self.random_process.reset_states()
    
    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )
        self.actor_target.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )
    
    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )