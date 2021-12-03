import torch
import torch.nn.functional as F

def triplet_loss(anchor, truthy, falsy, alpha=0.2):
    # input: unnormalised embeddings
    # output: triplet loss
    anchor = F.normalize(anchor)
    truthy = F.normalize(truthy)
    falsy = F.normalize(falsy)

    loss = torch.maximum((anchor - truthy).norm(dim=1, p=2)**2 - (anchor - falsy).norm(dim=1, p=2)**2 + alpha, torch.zeros(1).to(anchor.device))

    return loss.mean()