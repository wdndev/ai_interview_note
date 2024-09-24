import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import inspect
# from torch4keras.snippets import *


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """将序列padding到同一长度
    """
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)
    
    elif isinstance(inputs[0], torch.Tensor):
        assert mode == 'post', '"mode" argument must be "post" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


def get_kw(cls, kwargs, start_idx=3):
    '''保留类下的kwargs
    '''
    return {i:kwargs[i] for i in inspect.getargspec(cls)[0][start_idx:]}