import torch
import torch.nn as nn
import torch.autograd as autograd


class NTNHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(NTNHingeLoss, self).__init__()

        self.margin = margin

    def forward(self, yp, yn):
        yp = yp.repeat(yn.size())
        hinge = torch.clamp(self.margin + yp - yn, min=0.0)
        loss = torch.mean(hinge)
        return loss
