import  torch
from torch import nn

def loss_sic(align_logits, fuse_logits, targets):
    bs = targets.shape[0]
    loss_fn = nn.CrossEntropyLoss()

    labels = torch.arange(bs)

    loss_i = loss_fn(align_logits, labels)
    loss_s = loss_fn(align_logits.T, labels)
    loss_f = loss_fn(fuse_logits, targets)

    loss = (loss_i + loss_s)/2 + loss_f

    return loss

