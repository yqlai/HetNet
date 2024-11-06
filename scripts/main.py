from config import Config
from MBS import Macro_Base_Station
from CP import Content_Provider

import matplotlib.pyplot as plt

if __name__ == '__main__':
    config = Config()

    num_content = config.config['num_content']

    mbs = Macro_Base_Station(config)
    cp = [Content_Provider(config, i) for i in range(1, num_content+1)]

    time_slot = 1
    num_time_slot = config.config['num_time_slot']

    num_update = 0
    total_age = 0
    avg_arrival = sum([cp[i].arrival_rate * cp[i].num_user for i in range(num_content)])
    arr_AAoI = []

    while time_slot < num_time_slot:
        update_id = mbs.decide()
        for id in range(num_content):
            age, expired_user_request = cp[id].step(update_id)
            total_age += age * expired_user_request
        arr_AAoI.append(total_age / (avg_arrival * time_slot))

        if update_id > 0:
            num_update += 1

        time_slot += 1
    
    print('Update rate:', num_update / num_time_slot)

    plt.plot(range(1, num_time_slot), arr_AAoI)
    plt.show()