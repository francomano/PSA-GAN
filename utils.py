import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]    #the next has to be predicted
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def real_seq(data,seq_length):

    y=[]
    for i in range(seq_length,len(data)-seq_length-1):
        _y = data[i:(i+seq_length)]
        y.append(_y)
    return np.array(y)

    
def noise(Xt):
    bs = Xt.size(0)
    target_len = Xt.size(2)
    noise = torch.randn((bs, 1, target_len))
    Xt = torch.cat((Xt, noise), dim=1)
    return Xt


def create_folder(path):
    path = path
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    print("The new models directory is created!")
    return



def plot_training_history(name, discriminator_loss, generator_loss):
  fig, ax1 = plt.subplots(figsize=(20, 6))

  ax1.plot(discriminator_loss, 'o-', label='discriminator loss')
  ax1.plot(generator_loss, '^-', label='generator loss')

  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.legend()
  ax1.set_ylabel('Loss')
  ax1.set_xlabel('Epoch')

  fig.suptitle(f'Training history {name}')
  print("End-training Generator Loss:",generator_loss[-1])
  print("End-training Discriminator Loss:",discriminator_loss[-1])
  return