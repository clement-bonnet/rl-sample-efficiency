import os
import random

import numpy as np
import torch

def seed_python(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    
def seed_agent(random_seed, env):
    env.seed(random_seed)
    env.action_space.seed(random_seed)