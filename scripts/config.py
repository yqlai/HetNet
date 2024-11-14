import time
class Config:
    def __init__(self):
        self.config = {
            # System parameters
            'num_content': 1,
            'max_age': 50,
            'num_user': 3,
            'serve_offset': 6,
            'num_time_slot_in_episode': 1000,
            'num_episode': 15,
            'ref_state': [2, 1, 0, 1, 0, 1, 0, 1],
            'update_rate_upper_bound': 0.05,

            # Arrival parameters
            'kappa': 1.5,

            # DDPG
            'warmup':  200,
            'eta': 15,

            # Output
            'output_file': 'eta50.png'
        }
        # filename = 'MMddhhmm_{eta}_{serve_offset}_{num_content}.png'
        self.config['output_file'] = 'Result/' + time.strftime('%m%d%H%M') + '_' + str(self.config['eta']) + '_' + str(self.config['serve_offset']) + '_' + str(self.config['num_content']) + '.png'