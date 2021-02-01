import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
from utils import *
import random
from net import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MetaSGD(nn.Module):

    def __init__(self, outer_lr=1e-3):
        super(MetaSGD, self).__init__()
        self.net = Net()
        init_lr = random.uniform(5e-3, 0.1)

        self.w1_lr = Parameter(init_lr * torch.ones_like(self.net.w1, requires_grad=True))
        self.w2_lr = Parameter(init_lr * torch.ones_like(self.net.w2, requires_grad=True))
        self.w3_lr = Parameter(init_lr * torch.ones_like(self.net.w3, requires_grad=True))

        self.b1_lr = Parameter(init_lr * torch.ones_like(self.net.b1, requires_grad=True))
        self.b2_lr = Parameter(init_lr * torch.ones_like(self.net.b2, requires_grad=True))
        self.b3_lr = Parameter(init_lr * torch.ones_like(self.net.b3, requires_grad=True))

        self.task_lr = OrderedDict(
            [(name, p) for name, p in self.named_parameters() if 'lr' in name]
        )

        self.optimizer = optim.Adam(list(self.net.parameters()) + list(self.task_lr.values()), lr=outer_lr)

    def forward(self, k_x, k_y, q_x, q_y):
        task_num = k_x.size(0)
        losses = 0

        for i in range(task_num):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - lr * g) for (name, p), g, lr in zip(self.net.named_parameters(), grad, self.task_lr.values())]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])

            losses = losses + loss_q

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item() / task_num

    def test(self, k_x, k_y, q_x, q_y):
        load_state = torch.load('ckpt/meta_sgd.ckpt', map_location='cpu')
        self.load_state_dict(load_state['model_state_dict'])

        batch_size = k_x.size(0)
        losses = 0

        for i in range(batch_size):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - lr * g) for (name, p), g, lr in zip(self.net.named_parameters(), grad, self.task_lr.values())]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])
            losses = losses + loss_q

        losses /= batch_size

        return losses.item()

class MAML(nn.Module):

    def __init__(self, inner_lr, outer_lr):
        super(MAML, self).__init__()
        self.net = Net()
        self.inner_lr = inner_lr
        self.optimizer = optim.Adam(list(self.net.parameters()), lr=outer_lr)

    def forward(self, k_x, k_y, q_x, q_y):
        task_num = k_x.size(0)
        losses = 0

        for i in range(task_num):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])

            grad = torch.autograd.grad(loss_k, list(self.net.parameters()))

            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for (name, p), g in zip(self.net.named_parameters(), grad)]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])

            losses = losses + loss_q

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item() / task_num

    def test(self, k_x, k_y, q_x, q_y):
        load_state = torch.load('ckpt/maml.ckpt', map_location='cpu')
        self.load_state_dict(load_state['model_state_dict'])

        batch_size = k_x.size(0)
        losses = 0

        for i in range(batch_size):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for (name, p), g in zip(self.net.named_parameters(), grad)]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])
            losses = losses + loss_q

        losses /= batch_size

        return losses.item()






