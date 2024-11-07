import numpy as np

class Content_Provider:
    def __init__(self, config, id):
        self.id = id
        self.num_user = config.config['num_user']
        self.serve_offset = config.config['serve_offset']
        self.user_request_queue = [0] * (self.serve_offset+1)
        self.age = 0

        # Arrival rate = (id ^ -kappa) / sum(i ^ -kappa)
        self.arrival_rate = (id ** -config.config['kappa']) / sum([i ** -config.config['kappa'] for i in range(1, config.config['num_content']+1)])
        self.eta = config.config['eta']
    
    def initialize(self):
        for i in range(self.serve_offset+1):
            self.user_request_queue[i] = self.new_arrive()
        self.age = 1
    
    # Move the user request[1:delta] to [0:delta-1] and add new user request at the end
    # Add age according to the update flag
    # Return the age and the number of user requests
    def step(self, update_id):
        current_state = [self.age] + self.user_request_queue
        age, expired_user_request = self.age, self.user_request_queue[0]
        if update_id == self.id:
            update_indicator = 1
            self.age = 1
        else:
            update_indicator = 0
            self.age += 1
        
        self.user_request_queue = self.user_request_queue[1:] + [self.new_arrive()]
        next_state = [self.age] + self.user_request_queue
        
        return current_state, update_indicator, self.Lagrangian(age, expired_user_request, update_indicator)
    
    def new_arrive(self):
        return np.random.binomial(self.num_user, self.arrival_rate)

    def Lagrangian(self, age, user_request, update_indicator):
        return (age * user_request) / self.num_user * self.arrival_rate + self.eta * (update_indicator)