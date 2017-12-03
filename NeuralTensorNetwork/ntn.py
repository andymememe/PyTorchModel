import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class NeuralTensorNetwork(nn.Module):
    def __init__(self, inp_size=256, tsr_size=1, is_cuda=False):
        super(NeuralTensorNetwork, self).__init__()
        self.inp_size = inp_size
        self.is_cuda = is_cuda

        self.bilinear = nn.Bilinear(self.inp_size, self.inp_size, tsr_size)
        self.linear = nn.Linear(2 * self.inp_size, tsr_size)

    def forward(self, x1, x2):
        if self.is_cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
        biopt = self.bilinear(x1, x2)
        lopt = self.linear(torch.cat((x1, x2), 1))
        opt = biopt + lopt
        opt = F.tanh(opt)
        return opt
