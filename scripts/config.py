import time
class Config:
    def __init__(self):
        self.config = {
            # System parameters
            'num_content': 1,
            'max_age': 50,
            'num_user': 3,
            'serve_offset': 4,
            'num_time_slot_in_episode': 1000,
            'num_episode': 100,
            'ref_state': [2, 1, 0, 1, 0, 1, 0, 1, 0.1],
            'update_rate_upper_bound': 0.3,
            'num_time_slot_test': 10000,

            # Arrival parameters
            'kappa': 1.5,

            # DDPG
            'warmup':  200,
            'eta': 80,
            'learning_rate': 0.0001,

            # Output
            'output_file': 'eta50.png'
        }
        tmp_list = []
        for i in range(self.config['serve_offset']+1):
            tmp_list.append(i%3)
        self.config['ref_state'] = [2] + tmp_list + [0.1]
        self.config['output_file'] = '../Result/' + time.strftime('%m%d%H%M') + '_' + str(self.config['eta']) + '_' + str(self.config['serve_offset']) + '_' + str(self.config['num_content']) + '.png'