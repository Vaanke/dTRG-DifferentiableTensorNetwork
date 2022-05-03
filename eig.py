import numpy as np
import torch
from torch.functional import norm





def power(M):
    
    err = 0.1
    eps = 1e-8
    maxiter = 100000
    dim = M.shape
    v = torch.randn(dim[1],dtype=torch.float64)
    v = v/torch.norm(v)
    count = 0
    normM = torch.norm(M)
    M = M / normM
    while err > eps and count < maxiter:
        v2 = M @ v
        coef = torch.norm(v2)
        v2 = v2 / coef
        err0 = torch.norm(v - v2)
        err1 = torch.norm(v + v2)
        if err0 >= err1:
            lmd = - coef
            err = err1
            v2 = -v2
        else:
            lmd = coef
            err = err0
            v2 = v2
        v = v2
        count += 1
        # print(err)
    return v2, lmd*normM,  err


