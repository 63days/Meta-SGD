import torch
from sinusoid import Sinusoid
from meta import MetaSGD, MAML
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    print(device)
    if args.method == 'maml':
        meta = MAML(inner_lr=1e-2, outer_lr=args.outer_lr)
    else:
        meta = MetaSGD(outer_lr=args.outer_lr)

    meta.to(device)

    train_ds = Sinusoid(k_shot=args.k_shot, q_query=10, num_tasks=1000000)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_iter = iter(train_dl)

    losses = []

    pbar = tqdm(range(args.epochs))

    for epoch in pbar:
        k_i, k_o, q_i, q_o = next(train_iter)
        k_i, k_o, q_i, q_o = k_i.float().to(device), k_o.float().to(device), q_i.float().to(device), q_o.float().to(device)
        loss = meta(k_i, k_o, q_i, q_o)
        pbar.set_description(f'{epoch}/{args.epochs}iter | L:{loss:.4f}')

        if epoch % 100 == 0:
           losses.append(loss)

    torch.save({
        'model_state_dict': meta.state_dict(),
        'losses': losses
    }, f'ckpt/{args.method}.ckpt')

    plt.plot(losses)
    plt.savefig(f'results/{args.method}_loss_graph.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-SGD')

    parser.add_argument(
        '--method',
        type=str,
        default='maml',
        choices=['meta_sgd', 'maml']
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=60000
    )
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
        '--outer_lr',
        type=float,
        default=1e-3
    )
    args = parser.parse_args()

    main(args)
