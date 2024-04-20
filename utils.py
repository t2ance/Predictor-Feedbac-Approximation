import numpy as np
import torch


def count_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def padding_leading_zero(U, i, D_steps):
    start_index = max(i - D_steps, 0)
    end_index = i

    segment = U[start_index:end_index]

    if start_index < D_steps:
        if isinstance(U, np.ndarray):
            padding = np.zeros(D_steps - len(segment))
            segment = np.concatenate((padding, segment))
        elif isinstance(U, torch.Tensor):
            padding = torch.zeros(D_steps - len(segment))
            segment = torch.concatenate((padding, segment))

    return segment


if __name__ == '__main__':
    ...
