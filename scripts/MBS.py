import numpy as np

class Macro_Base_Station: 
    def __init__(self, config):
        self.num_content = config.config['num_content']
    
    def decide(self):
        return self.random_decide()
    
    def random_decide(self, update_rate: float):
        if np.random.rand() < update_rate:
            return 1
        else:
            return 0