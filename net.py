import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
from utils import truncated_normal
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.w1 = Parameter(truncated_normal(1, 40, std=1e-2), requires_grad=True)
        self.w2 = Parameter(truncated_normal(40, 40, std=1e-2), requires_grad=True)
        self.w3 = Parameter(truncated_normal(40, 1, std=1e-2), requires_grad=True)

        self.b1 = Parameter(torch.zeros(40), requires_grad=True)
        self.b2 = Parameter(torch.zeros(40), requires_grad=True)
        self.b3 = Parameter(torch.zeros(1), requires_grad=True)

        self.params = OrderedDict([('w1', self.w1), ('w2', self.w2), ('w3', self.w3),
                                   ('b1', self.b1), ('b2', self.b2), ('b3', self.b3)])

    def forward(self, x, weights=None):
        if weights is None:
            weights = OrderedDict([(name, p) for name, p in self.named_parameters()])

        x = F.relu(x.matmul(weights['w1']) + weights['b1'], inplace=True)
        x = F.relu(x.matmul(weights['w2']) + weights['b2'], inplace=True)
        x = x.matmul(weights['w3']) + weights['b3']

        return x


