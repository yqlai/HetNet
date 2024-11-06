import numpy as np

class Macro_Base_Station: 
    def __init__(self, config):
        self.num_content = config.config['num_content']
    
    def decide(self):
        return np.random.randint(0, self.num_content+1)