import numpy as np

# Chunk a batch of sequences.
# Inputs : Tensor of shape (batch_size, time_steps, dimension),
#           left_context, mid_context, right_context
# Outputs : Chunked input tensor, original time
# Using strict chunking avoids the assumption that the sequence is circular,
# but impacts the performance for longer length sequences by a slight margin.
def chunk(X, left, mid, right, strict=False):
    was_unsqueezed = False
    if left > mid or right > mid:
        return _chunk_extended(X, left, mid, right, strict=strict)
    # Handling single dimensional inputs which are squeezed.
    if len(X.shape) == 2:
        X = np.expand_dims(X, 2)
        was_unsqueezed = True
    N, t, d = X.shape
    n_chunks = int(np.ceil(float(t) / mid))
    t_new = n_chunks * mid
    t_extra = t_new - t
    X = np.pad(X, ((0, 0), (0, t_extra), (0, 0)), mode='constant', constant_values=0)
    X = X.reshape((N, n_chunks, mid, d))
    # Padding zeros on the chunks to avoid circular sequence assumption.
    if strict:
        X = np.pad(X, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    X_left = X[:, :, mid-left:, :]
    X_left = np.roll(X_left, 1, 1)
    X_right = X[:, :, :right, :]
    X_right = np.roll(X_right, -1, 1)
    X = np.concatenate([X_left, X, X_right], axis=2)
    if strict:
        X = X[:, 1:-1, :, :]
    if was_unsqueezed:
        X = X.squeeze(3)
    return X, t

# Extended support for chunking when left or right > mid.
# Not using this by default because using lists worsen the performance by a small amount.
def _chunk_extended(X, left, mid, right, strict=False):
    # Handling single dimensional inputs which are squeezed.
    was_unsqueezed = False
    if len(X.shape) == 2:
        X = np.expand_dims(X, 2)
        was_unsqueezed = True
    N, t, d = X.shape
    n_chunks = int(np.ceil(float(t) / mid))
    t_new = n_chunks * mid
    t_extra = t_new - t
    X = np.pad(X, ((0, 0), (0, t_extra), (0, 0)), mode='constant', constant_values=0)
    X = X.reshape((N, n_chunks, mid, d))
    left_reps = int(np.ceil(float(left) / mid))
    right_reps = int(np.ceil(float(right) / mid))
    if strict:
        X = np.pad(X, ((0, 0), (left_reps, right_reps), (0, 0), (0, 0)), mode='constant', constant_values=0)
    X_list = []
    for roll in range(left_reps, -(right_reps+1), -1):
        X_list.append(np.roll(X, roll, 1))
    X = np.concatenate(X_list, axis=2)
    zero_index = mid * left_reps
    X = X[:, :, zero_index - left: zero_index + mid + right, :]
    if strict:
        X = X[:, left_reps:n_chunks+left_reps, :, :]
    if was_unsqueezed:
        X = X.squeeze(3)
    return X, t

# Inverse of the chunking operation.
# Inputs : chunked tensor from chunk() function, left_context, mid_context, right_context, original_time
# Output : Tensor of shape same as input to the chunk() function
def unchunk(X, left, mid, right, t_original):
    # Handling single dimensional inputs which are squeezed.
    was_unsqueezed = False
    if len(X.shape) == 3:
        X = np.expand_dims(X, 3)
        was_unsqueezed = True
    N, n_chunks, size_chunk, d = X.shape
    X = X[:, :, left:left+mid, :]
    X = X.reshape((N, n_chunks * mid, d))[:, :t_original, :]
    if was_unsqueezed:
        X = X.squeeze(2)
    return X
