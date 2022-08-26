from torchvision.datasets import MNIST
import numpy as np
import math
import time
from contextlib import contextmanager


RNG = np.random.default_rng()


# TODO: change into a generator
def batch_data(data_size, batch_size):
    p = RNG.permutation(data_size)
    num_of_batches = math.floor(data_size / batch_size)
    data_batches = [p[i*batch_size:(i+1)*batch_size] for i in range(num_of_batches)]
    if num_of_batches * batch_size < data_size:
        data_batches.append(p[num_of_batches * batch_size:])
    return data_batches


def random_split(split_size, data_size):
    p = RNG.permutation(data_size)
    return p[:split_size], p[split_size:]


def get_simple_data(size):
    images = RNG.integers(low=0, high=3, size=(size, 2))
    labels = RNG.integers(low=0, high=4, size=size)
    return images, labels


def get_data(size, train=True):
    try:
        data = MNIST(root='data/', train=train)
    except RuntimeError:
        data = MNIST(root='data/', train=train, download=True)
    images = data.data.numpy()
    images = images.reshape(images.shape[0], -1)
    labels = data.targets.numpy()
    p, _ = random_split(size, len(data))
    return images[p], labels[p]


def l2_distance(ndarr1, ndarr2):
    return np.linalg.norm(ndarr1 - ndarr2, axis=-1)
    # return np.sqrt(((ndarr1 - ndarr2) ** 2).sum(axis=-1))


def squared_l2_distance(ndarr1, ndarr2):
    return np.sum((ndarr1 - ndarr2)**2, axis=-1)
    #  return ((ndarr1 - ndarr2) ** 2).sum(axis=-1)


# Time utilities

def ping():
    """Generate a time value to be later used by pong."""
    return time.time() * 1000


def pong(ping_, ms_rounding=3):
    """Returns the time delta in ms."""
    return round((time.time() * 1000) - ping_, ms_rounding)


@contextmanager
def pingpong(desc='Pingpong', show=True, return_elapsed=None, ms_rounding=3):
    """A context manager to record elapsed time of execution of a code block."""
    p = ping()
    yield p
    elapsed = pong(p, ms_rounding)
    if show:
        print(f'{desc} elapsed in {elapsed}ms')
    if callable(return_elapsed):
        return_elapsed(elapsed)


