import time
import numpy as np
import torch
import torch.nn as nn

from MBS import Macro_Base_Station
from CP import Content_Provider
from utils import *
from ddpg import DDPG

def train(mbs:Macro_Base_Station, arr_cp:list[Content_Provider], config):
    print('Start training...')
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sum_AoI = 0
    arr_AAoI = []
    num_update = 0

    for cp in arr_cp:
        cp.initialize()

        cp.agent.is_training = True
        episode = episode_slot = 0
        num_episode = config.config['num_episode'] # 200
        num_time_slot_in_episode = config.config['num_time_slot_in_episode'] # 1000
        episode_lagrangian = 0
        episode_threshold = 0
        while episode < num_episode:
            sum_AoI += cp.age * cp.user_request_queue[0]
            arr_AAoI.append(sum_AoI / ((episode * num_time_slot_in_episode + episode_slot + 1) * cp.arrival_rate))

            current_state = [cp.age] + cp.user_request_queue
            if episode > 0 or episode_slot > config.config['warmup']:
                update_flag, actions = cp.agent.select_action(current_state, False)
                threshold = torch.argmax(actions).item() + 1
                episode_threshold += threshold
            else:
                update_flag, actions = cp.agent.random_action(current_state)
                threshold = torch.argmax(to_tensor(actions)).item() + 1
                episode_threshold += threshold
                
            if update_flag == 0:
                update_id = 0
            else:
                update_id = cp.id
                num_update += 1

            next_state, update_indicator, Lagrangian = cp.step(update_id)

            cp.agent.observe(Lagrangian, next_state)
            if episode > 0 or episode_slot > config.config['warmup']:
                cp.agent.update_policy()
            
            episode_slot += 1
            episode_lagrangian += Lagrangian
            if episode_slot == num_time_slot_in_episode:
                avg_Lagrangian = episode_lagrangian / num_time_slot_in_episode
                avg_threshold = episode_threshold / num_time_slot_in_episode
                print(f'Episode: {episode}, Average Lagrangian: {avg_Lagrangian}, Average Threshold: {avg_threshold}')

                episode += 1
                episode_lagrangian = 0
                episode_threshold = 0
                episode_slot = 0
    
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    
    return arr_AAoI, num_update / (num_episode * num_time_slot_in_episode)