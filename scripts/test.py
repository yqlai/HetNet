import time
import numpy as np
import torch
import torch.nn as nn

from MBS import Macro_Base_Station
from CP import Content_Provider
from utils import *

def test(mbs: Macro_Base_Station, arr_cp: list[Content_Provider], config):
    num_time_slot = config.config['num_time_slot_test']

    for cp in arr_cp:
        cp.agent.is_training = False
        cp.initialize()
    
    sum_AoI = 0
    arr_AAoI = []
    num_update = 0

    for time_slot in range(1, num_time_slot+1):

        sum_AoI += sum([cp.age * cp.user_request_queue[0] for cp in arr_cp])
        arr_AAoI.append(sum_AoI / (time_slot * sum([cp.arrival_rate * cp.num_user for cp in arr_cp])))
        
        # Calculate the sum of Value function for each action
        arr_value = [0] * (len(arr_cp)+1)
        for action in range(len(arr_cp)+1):
            arr_value[action] = 0
            
            for cp in arr_cp:
                current_state = [cp.age] + cp.user_request_queue + [cp.num_update/max(cp.num_time_slot, 1)]
                update_indicator = 1 if action == cp.id else 0
                value = cp.ddpg.critic(to_tensor(current_state), to_tensor([update_indicator]))
                arr_value[action] += value.item()
        
        # Select the action with the maximum value
        action = np.argmax(arr_value)
        if action > 0:
            num_update += 1

        # Update the system
        for cp in arr_cp:
            cp.step(action)
