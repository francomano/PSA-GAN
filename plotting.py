
import torch
import utils
import metrics
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_dataset(data, x_axis, y_axis):
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.plot(data)
    plt.show()


def plot_training_history(name, discriminator_loss, generator_loss):
    fig, ax1 = plt.subplots(figsize=(18, 5))
    
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

def plot_fid_history(name, fid):
    fig, ax1 = plt.subplots(figsize=(18, 5))
    ax1.plot(fid, 'o-', label='Context-FID score')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend()
    ax1.set_ylabel('Context-FID')
    ax1.set_xlabel('Epoch')

    fig.suptitle(f'Training history {name}')
    ax1.set_xticks(range(1, len(fid) + 1))
    print("End-training Context-FID:",fid[-1])
    return

def plot_prediction(fake_data, real_data, title, name_model):
    plt.figure(figsize=(7, 3), dpi=300)
    
    plt.plot(fake_data, label='Generated series', color="red")
    plt.plot(real_data, label='Actual series', )

    #plt.suptitle(f'Train set: {name_model}')
    plt.suptitle(title+name_model)
    plt.legend()
    plt.show()



def prediction_test(seq_length, batch_size, test_set, testX, G, name_model):
    seq_length = 512 #it is tau in the paper

    batch_size=121

    t = test_set[len(testX):len(testX)+seq_length]

    with torch.no_grad():
        
        generated_series = G(testX[len(testX)-batch_size:],1,(G.step-1)) 
        generated_series=generated_series.permute(0,2,1)
        #print("generated_series: ",generated_series.shape)
        generated_series = generated_series.to("cpu").detach().numpy()

    generated_series[batch_size-1]=utils.scale(generated_series[batch_size-1])
    aux=utils.scale(t)

    nrmse=metrics.nrmse(torch.Tensor(generated_series), torch.Tensor(aux))
    print("NRMSE on test set: ",nrmse)
    plot_prediction(generated_series[batch_size-1], aux, "Test set: ", name_model)
