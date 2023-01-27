import numpy as np
import torch

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]    #the next has to be predicted
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def real_seq(datay,seq_length):
    datay=datay.detach().numpy()
    y=[]
    for i in range(seq_length,len(datay)-seq_length-1):
        _y = datay[i:(i+seq_length)]
        y.append(_y)


    return np.array(y)
def noise(Xt):
    bs = Xt.size(0)
    target_len = Xt.size(2)
    noise = torch.randn((bs, 1, target_len))
    Xt = torch.cat((Xt, noise), dim=1)
    return Xt