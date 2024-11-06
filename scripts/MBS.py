import numpy as np

class Macro_Base_Station: 
    def __init__(self, config):
        self.num_content = config.config['num_content']
    
    def decide(self):
        # choose [0, 1, 2] with prob [0.8, 0.1, 0.1]
        if np.random.rand() < 0.8:
            return 0
        else:
            return np.random.randint(1, self.num_content+1)