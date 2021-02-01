import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from sinusoid import Sinusoid
from meta import MetaSGD, MAML

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    maml = MAML(inner_lr=args.inner_lr, outer_lr=args.outer_lr)
    meta_sgd = MetaSGD(outer_lr=args.outer_lr)

    test_ds = Sinusoid(k_shot=args.k_shot, q_query=100, num_tasks=100)
    test_dl = DataLoader(test_ds, batch_size=100, pin_memory=True)
    test_iter = iter(test_dl)

    k_x, k_y, q_x, q_y = next(test_iter)
    k_x, k_y, q_x, q_y = k_x.float(), k_y.float(), q_x.float(), q_y.float()

    loss_meta_sgd = meta_sgd.test(k_x, k_y, q_x, q_y)
    loss_maml = maml.test(k_x, k_y, q_x, q_y)

    print(loss_meta_sgd, loss_maml)

    amplitude = np.random.uniform(0.1, 5.0, size=1)
    phase = np.random.uniform(0., np.pi, size=1)

    xs = np.linspace(-5, 5).reshape(-1, 1).astype('float32')
    ys = (amplitude * np.sin(xs + phase)).astype('float32')

    idx = np.random.choice(50, args.k_shot, replace=True)
    k_x, k_y = torch.from_numpy(xs[idx]), torch.from_numpy(ys[idx])
    xs, ys = torch.from_numpy(xs), torch.from_numpy(ys)

    ################# Meta-SGD #####################
    pre_pred_sgd = meta_sgd.net(xs).cpu().detach().numpy()
    pred_k = meta_sgd.net(k_x)
    loss_k = F.mse_loss(pred_k, k_y)
    grad = torch.autograd.grad(loss_k, meta_sgd.net.parameters())
    fast_weights = OrderedDict(
        [(name, p - lr * g) for (name, p), g, lr in zip(meta_sgd.net.named_parameters(), grad, meta_sgd.task_lr.values())]
    )
    pred_sgd = meta_sgd.net(xs, fast_weights)
    pred_sgd = pred_sgd.cpu().detach().numpy()
    #################################################

    #################   MAML   ######################
    pre_pred_maml = maml.net(xs).cpu().detach().numpy()
    pred_k = maml.net(k_x)
    loss_k = F.mse_loss(pred_k, k_y)
    grad = torch.autograd.grad(loss_k, maml.net.parameters())
    fast_weights = OrderedDict(
        [(name, p - maml.inner_lr * g) for (name, p), g in zip(maml.net.named_parameters(), grad)]
    )
    pred_maml = maml.net(xs, fast_weights)
    pred_maml = pred_maml.cpu().detach().numpy()

    plt.plot(xs, ys, 'gray', linewidth=2.3, label='Ground Truth')
    plt.plot(xs, pred_sgd, 'r', linewidth=2.3, label='Meta-SGD')
    plt.plot(xs, pred_maml, 'c', linewidth=2.3, label='MAML')

    plt.plot(xs, pre_pred_sgd, 'r-.', linewidth=0.7)
    plt.plot(xs, pre_pred_maml, 'c-.', linewidth=0.7)

    plt.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-SGD')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--k_shot',
        type=int,
        default=10
    )
    parser.add_argument(
        '--inner_lr',
        type=float,
        default=1e-2
    )
    parser.add_argument(
        '--outer_lr',
        type=float,
        default=1e-3
    )
    args = parser.parse_args()

    main(args)