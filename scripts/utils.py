import torch
from torch.autograd import Variable
import time

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT, device=None):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype).to(device)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
def saveconfig(config, filename, update_rate):
    with open(filename, 'w') as f:
        # Date and Time
        f.write('Date:%s\n' % time.strftime('%m/%d/%Y %H:%M:%S'))
        for key, value in config.items():
            f.write('%s:%s\n' % (key, value))
        if type(update_rate) == list:
            for i in update_rate:
                f.write('Update Rate:%s\n' % i)
        else:
            f.write('Update Rate:%s\n' % update_rate)