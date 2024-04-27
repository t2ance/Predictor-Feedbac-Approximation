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


def pad_leading_zeros(segment, length):
    assert len(segment) <= length

    if len(segment) < length:
        padding_length = length - len(segment)
        if isinstance(segment, np.ndarray):
            padding = np.zeros(padding_length)
            segment = np.concatenate((padding, segment))
        elif isinstance(segment, torch.Tensor):
            padding = torch.zeros(padding_length)
            segment = torch.concatenate((padding, segment))

    return segment


if __name__ == '__main__':
    ...
