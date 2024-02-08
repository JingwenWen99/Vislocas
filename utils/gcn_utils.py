import numpy as np
import pandas as pd

import torch

def gen_A(adj_file):
    adj = pd.read_csv(adj_file, header=0, index_col=0)
    _adj = adj.values
    _nums = np.diag(_adj)[:, np.newaxis]
    _adj = _adj / (_nums + 1e-6)
    # _adj[_adj < t] = 0
    return _adj


def gen_adj(A):
    D = torch.pow((A.sum(1).float() + 1e-6), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


if __name__ == '__main__':
    _adj = gen_A(10, 0.1, "")
    from torch.nn import Parameter
    A = Parameter(torch.from_numpy(_adj).float())
    # print(A)
    adj = gen_adj(A).detach()
    print(adj)