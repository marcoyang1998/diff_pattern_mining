import matplotlib.pyplot as plt


def draw_trend():
    pass


def to_device(data, device):
    if isinstance(data, list):
        for i, x in enumerate(data):
            data[i] = x.to(device)

    pass