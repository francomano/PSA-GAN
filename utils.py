import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

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

    
def noise(Xt, device):
    bs = Xt.size(0)
    target_len = Xt.size(2)
    noise = torch.randn((bs, 1, target_len))
    noise=noise.to(device)
    Xt=Xt.to(device)
    Xt = torch.cat((Xt, noise), dim=1)
    return Xt


def create_folder(path):
    path = path
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    print("New models directory created!")
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
  ax1.set_xticks(range(1, len(generator_loss) + 1))
  print("End-training Generator Loss:",generator_loss[-1])
  print("End-training Discriminator Loss:",discriminator_loss[-1])
  return


def scale(generated):
    sc = MinMaxScaler()
    generated = sc.fit_transform(generated)
    return generated


def assign_device(gpu):
    if (torch.cuda.is_available() and gpu==True):
        device = "cuda"
        print("Cuda enabled: using GPU")
    else:
        device = "cpu"
        print("Cuda not available: using CPU")
    return device


def moment_loss(fake_data, real_data):
    fake_mean = fake_data.mean()
    real_mean = real_data.mean()
    fake_std = fake_data.std()
    real_std = real_data.std()
    return abs(fake_mean - real_mean) + abs(fake_std - real_std)


def mse(fake_data, real_data):
    return np.mean(np.square(real_data-fake_data))


def abs_error(fake_data, real_data):
    return np.sum(np.abs(real_data-fake_data))


def nrmse(fake_data, real_data):
    return np.sqrt(np.mean((real_data-fake_data)**2)) / (np.max(real_data)-np.min((real_data)))


from tqdm import tqdm
def evaluation(eval, test_loader, G):
    
    results=[]
    with torch.no_grad():
        device='cuda'
        G.to(device)
        for i, (X, Y) in tqdm(enumerate((test_loader)), total=len(test_loader)):
            #print(i)
            if(i==135):
                break
                print("Critic point")
            X=X.to(device)
            Y=Y.to(device)
            
            fake_data = G(X,1,(G.step-1))
            real_data=Y[:,:,:fake_data.size(2)].to("cpu")

            fake_data=fake_data.permute(0,2,1)
            fake_data = fake_data.to("cpu").detach().numpy()
            real_data = real_data.to("cpu").detach().numpy()

            #fake_data=utils.scale(fake_data)
            #real_data=utils.scale(real_data)

            if(eval=="NRMSE"):
                result=nrmse(fake_data, real_data)
                results.append(result)
            elif(eval=="MSE"):
                result=mse(fake_data, real_data)
                results.append(result)
            elif(eval=="ABS"):
                result=abs_error(fake_data, real_data)
                results.append(result)
            else:
                print("error")
                break

    res_mean=np.mean(results)
    res_stdev=np.std(results)

    return res_mean, res_stdev