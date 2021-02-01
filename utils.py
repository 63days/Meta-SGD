import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def truncated_normal(*size, mean=0, std=1):
    size = list(size)
    tensor = torch.empty(size)
    tmp = torch.empty(size+[4,]).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor