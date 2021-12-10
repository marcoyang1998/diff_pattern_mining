import matplotlib.pyplot as plt
import torch.nn as nn

def draw_trend():
    pass

def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)