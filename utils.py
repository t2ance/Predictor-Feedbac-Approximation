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


def pad_leading_zeros(U, start, end):
    assert start <= end
    r = end - start
    start_index = max(start, 0)
    end_index = end

    segment = U[start_index:end_index]

    if start_index < r:
        if isinstance(U, np.ndarray):
            padding = np.zeros(r - len(segment))
            segment = np.concatenate((padding, segment))
        elif isinstance(U, torch.Tensor):
            padding = torch.zeros(r - len(segment))
            segment = torch.concatenate((padding, segment))

    return segment


if __name__ == '__main__':
    ...
