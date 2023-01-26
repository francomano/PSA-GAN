import numpy as np
import torch


def sliding_windows(data, seq_length):          
    x = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        x.append(_x)
    return np.array(x)



def get_batch(data,seq_length,num_samples,index):

    x = []

    for i in range(num_samples):
        _x = data[i+index:(i+index+seq_length)]
        x.append(_x)
    return np.array(x)