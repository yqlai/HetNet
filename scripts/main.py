from config import Config
from MBS import Macro_Base_Station
from CP import Content_Provider
from train import train

import matplotlib.pyplot as plt

def random_method(mbs: Macro_Base_Station, arr_cp:Content_Provider, config, update_rate: float):
    for cp in arr_cp:
        cp.initialize()
    num_update = 0
    time_slot = 1

    while time_slot < config.config['num_episode'] * config.config['num_time_slot_in_episode']:
        update_id = mbs.random_decide(update_rate)
        for cp in arr_cp:
            cp.step(update_id)
        if update_id > 0:
            num_update += 1
        time_slot += 1




if __name__ == '__main__':
    config = Config()

    num_content = config.config['num_content']

    mbs = Macro_Base_Station(config)
    cp = [Content_Provider(config, i) for i in range(1, num_content+1)]

    arr_AAoI_ddpg, update_rate = train(mbs, cp, config)
    

    plt.plot(range(len(arr_AAoI_ddpg)), arr_AAoI_ddpg)
    plt.savefig('result.png')
    plt.show()

    print('Update rate:', update_rate)

    # time_slot = 1
    # num_time_slot = config.config['num_time_slot']

    # num_update = 0
    # total_age = 0
    # avg_arrival = sum([cp[i].arrival_rate * cp[i].num_user for i in range(num_content)])
    # arr_AAoI = []

    # while time_slot < num_time_slot:
    #     update_id = mbs.decide()
    #     for id in range(num_content):
    #         age, expired_user_request = cp[id].step(update_id)
    #         total_age += age * expired_user_request
    #     arr_AAoI.append(total_age / (avg_arrival * time_slot))

    #     if update_id > 0:
    #         num_update += 1

    #     time_slot += 1
    
    # print('Update rate:', num_update / num_time_slot)

    # plt.plot(range(1, num_time_slot), arr_AAoI)
    # plt.show()