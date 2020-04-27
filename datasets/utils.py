import torch
import numpy as np


def normalize(data, norm_param):
    """Normalize each data sample independently."""
    if 'mean' in norm_param:
        data = data - torch.mean(data, dim=0, keepdim=True)
    if 'l2' in norm_param:
        data = data / torch.norm(data, dim=1, keepdim=True)
    return data

def random_mask(train_set, test_set, n_feat):
    features = np.random.choice(range(train_set.shape[1]), n_feat, replace=False)
    train_set = train_set[:, features]
    test_set = test_set[:, features]
    return train_set, test_set
