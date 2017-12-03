import torch
import torch.nn as nn
import torch.autograd as autograd


class Autoencoder(nn.Module):
    def __init__(self, inp_size=300, hid_size=256, vocab_size=927, is_cuda=False):
        super(Autoencoder, self).__init__()

        self.inp_size = inp_size
        self.hid_size = hid_size
        self.is_cuda = is_cuda

        self.word_embeddings = nn.Embedding(vocab_size, self.inp_size)
        self.lstm = nn.LSTM(input_size=self.inp_size, hidden_size=self.hid_size, num_layers=1)
        self.linear = nn.Linear(self.hid_size, self.inp_size)

        self.hidden = self.init_hidden(self.hid_size)

    def import_embedding(self, w):
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(w))
        self.word_embeddings.weight.requires_grad = False

    def init_hidden(self, hid_size):
        if self.is_cuda:
            return (autograd.Variable(torch.zeros(1, 1, hid_size)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, hid_size)).cuda())
        return (autograd.Variable(torch.zeros(1, 1, hid_size)),
                autograd.Variable(torch.zeros(1, 1, hid_size)))

    def forward(self, inp):
        embeds = self.word_embeddings(inp)
        if self.is_cuda:
            embed = embed.cuda()
        opt, hid = self.lstm(embeds.view(len(inp), 1, -1), self.hidden)
        opt = self.linear(opt)
        return opt, embeds, hid[0].view(1, self.hid_size)
