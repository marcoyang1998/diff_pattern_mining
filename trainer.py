import torch.nn as nn
from torch import optim
from losses import triplet_loss
from model import BasePatternMiner, get_model


class BaseTrainer(nn.Module):
    def __init__(self, model: BasePatternMiner, args, device='cpu'):
        super(BaseTrainer, self).__init__()
        self.optim = optim.SGD(model.parameters(), lr=args.lr)
        self.model = model
        self.device = device
        self.model.to(device)

    def update(self, x, *args, **kwargs):
        self.model.clamp_weight()
        raise NotImplementedError('Update method is not implemented')

    def get_pattern(self):
        return self.model.get_pattern()

    def print_pattern(self):
        raise NotImplementedError()


class PatterMiningClassificationTrainer(BaseTrainer):
    def __init__(self, model: BasePatternMiner, args, device='cpu'):
        super(PatterMiningClassificationTrainer, self).__init__(model, args, device)
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

    def print_pattern(self):
        pattern, weight = self.get_pattern()
        pattern = pattern.cpu().detach().numpy()
        weight = weight.view(-1).cpu().detach().numpy()
        num_dim = pattern.shape[0]
        for i in range(num_dim):
            print(f"Pattern {i + 1} with weight {weight[i]}:\n {pattern[i]}")


class PatternMiningContrastiveTrainer(BaseTrainer):
    def __init__(self, model: BasePatternMiner, args, device='cpu'):
        super(PatternMiningContrastiveTrainer, self).__init__(model, device)
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

        return loss

    def print_pattern(self):
        pattern = self.get_pattern()
        num_dim = pattern.shape[0]
        for i in range(num_dim):
            print(f"Pattern {i + 1}:\n {pattern[i]}")


def get_trainer(device, args, **kwargs) -> BaseTrainer:
    model = get_model(args, **kwargs)
    if args.loss_type == 'classification':
        return PatterMiningClassificationTrainer(model=model, device=device, args=args)
    elif args.loss_type == 'contrastive':
        return PatternMiningContrastiveTrainer(model=model, device=device, args=args)
    else:
        raise NotImplementedError(f'Not implemented: {args.loss_type}')

