# Custom functions for pytorch.
__author__ = "Kiran Praveen"
__version__ = "1.0"
__email__ = "kirant1994@gmail.com"

import numpy as np
import torch

def _save_model(net, path):
    torch.save(net.state_dict(), path)

def _load_model(net, path):
    net.load_state_dict(torch.load(path))

# Inherited Module to include load and save.
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        _save_model(self, path)

    def load(self, path):
        _load_model(self, path)

# Inherited DataParallel to include load and save.
class DataParallel(torch.nn.DataParallel):
    def __init__(self, net):
        super().__init__(net)

    def save(self, path):
        _save_model(self.module, path)

    def load(self, path):
        _load_model(self.module, path)

"""
Chunk a batch of sequences.
Inputs : Tensor of shape (batch_size, time_steps, dimension),
		 left_context, mid_context, right_context
Outputs : Chunked input tensor, original time
Using strict chunking avoids the assumption that the sequence is circular,
but impacts the performance for longer length sequences by a slight margin.
""" 
def chunk(X, left, mid, right, strict=False):
    if left > mid or right > mid:
        return _chunk_extended(X, left, mid, right, strict=strict)
    N, t, d = X.shape
    n_chunks = int(np.ceil(float(t) / mid))
    t_new = n_chunks * mid
    t_extra = t_new - t
    X = torch.nn.functional.pad(X, pad=(0, 0, 0, t_extra), mode='constant', value=0)
    X = X.view((N, n_chunks, mid, d))
    # Padding zeros on the chunks to avoid circular sequence assumption.
    if strict:
        X = torch.nn.functional.pad(X, pad=(0, 0, 0, 0, 1, 1), mode='constant', value=0)
    X_left = X[:, :, mid-left:, :]
    X_left = torch.roll(X_left, 1, 1)
    X_right = X[:, :, :right, :]
    X_right = torch.roll(X_right, -1, 1)
    X = torch.cat([X_left, X, X_right], dim=2)
    if strict:
        X = X[:, 1:-1, :, :].contiguous()
    return X, t

"""
Extended support for chunking when left or right > mid.
Not using this by default because using lists worsen the performance by a small amount.
"""
def _chunk_extended(X, left, mid, right, strict=False):
    N, t, d = X.shape
    n_chunks = int(np.ceil(float(t) / mid))
    t_new = n_chunks * mid
    t_extra = t_new - t
    X = torch.nn.functional.pad(X, pad=(0, 0, 0, t_extra), mode='constant', value=0)
    X = X.view((N, n_chunks, mid, d))
    left_reps = int(np.ceil(float(left) / mid))
    right_reps = int(np.ceil(float(right) / mid))
    if strict:
    	X = torch.nn.functional.pad(X, pad=(0, 0, 0, 0, left_reps, right_reps), mode='constant', value=0)
    X_list = []
    for roll in range(left_reps, -(right_reps+1), -1):
        X_list.append(torch.roll(X, roll, 1))
    X = torch.cat(X_list, dim=2)
    zero_index = mid * left_reps
    X = X[:, :, zero_index - left: zero_index + mid + right, :]
    if strict:
    	X = X[:, left_reps:n_chunks+left_reps, :, :]
    return X, t

"""
Inverse of the chunking operation.
Inputs : chunked tensor from chunk() function, left_context, mid_context, right_context, original_time
Output : Tensor of shape same as input to the chunk() function
"""
def unchunk(X, left, mid, right, t_original):
    N, n_chunks, size_chunk, d = X.shape
    X = X[:, :, left:left+mid, :].contiguous()
    X = X.view((N, n_chunks * mid, d))[:, :t_original, :]
    return X
