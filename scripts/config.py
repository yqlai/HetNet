class Config:
    def __init__(self):
        self.config = {
            # System parameters
            'num_content': 1,
            'max_age': 50,
            'num_user': 3,
            'serve_offset': 3,
            'num_time_slot_in_episode': 1000,
            'num_episode': 200,
            'ref_state': [2, 1, 0, 1, 0],

            # Arrival parameters
            'kappa': 1.5,

            # DDPG
            'warmup': 200,
            'eta': 15
        }