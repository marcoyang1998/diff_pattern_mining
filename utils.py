import matplotlib.pyplot as plt
import torch


def draw_trend():
    pass


def pattern_init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        with torch.no_grad():
            m.weight[m.weight < 0] = 0
        m.bias.data.fill_(0.0)


def to_device(data, device):
    if isinstance(data, list):
        for i, x in enumerate(data):
            data[i] = x.to(device)

    pass