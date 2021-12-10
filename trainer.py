import torch.nn as nn
from torch import optim
from losses import triplet_loss
from model import BasePatternMiner

class BaseTrainer(nn.Module):
    def __init__(self, model: BasePatternMiner, device='cpu'):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(device)

    def update(self, x, *args, **kwargs):
        raise NotImplementedError('Update method is not implemented')


class PatterMiningClassifier(BaseTrainer):
    def __init__(self, model: BasePatternMiner, args, device='cpu'):
        super(PatterMiningClassifier, self).__init__(model, device)
        self.epoch = 0
        self.optim = optim.SGD(model.parameters(), lr=args.lr)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x.to(self.device)
        out = self.model(x).squeeze(dim=-1)
        return out

    def update(self, x, label):
        label.to(self.device)
        out = self.forward(x)
        loss = self.loss(out, label.float())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def get_pattern(self):
        return self.model.get_pattern()


class PatternMiningContrastiveTrainer(BaseTrainer):
    def __init__(self, model: BasePatternMiner, args, device='cpu'):
        super(PatternMiningContrastiveTrainer, self).__init__(model, device)
        self.optim = optim.SGD(model.parameters(), lr=args.lr)
        self.loss = triplet_loss

    def forward(self, anchor, truthy, falsy):
        anchor = self.model(anchor)
        truthy = self.model(truthy)
        falsy = self.model(falsy)

        return anchor, truthy, falsy

    def update(self, anchor, truthy, falsy):
        loss = self.loss(*self.forward(anchor, truthy, falsy))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.model.clamp_weight()

        return loss

    def get_pattern(self):
        return self.model.get_pattern()