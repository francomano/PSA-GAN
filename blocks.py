import numpy as np
import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


        ################################################
        #self.nb_step = int(math.log2(tau)) - 2
        #F.avg_pool1d(
        #    x, kernel_size=2 ** (self.nb_step - 1)
        #) 
        #############################################

class pool(nn.Module):
    def __init__(self, tau):
        #print("tau: ",tau)
        super().__init__()
        str=(math.log2(tau) - 3)
        kernel=int(2**str)
        #print("str: ",str)
        self.pool=nn.AvgPool1d(kernel_size=kernel)
    def forward(self,x):
        z=self.pool(x)
        return z



class Self_Attn(nn.Module):
   
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
        
    def forward(self,x):
        
        
        query  = self.query_conv(x)
        key =  self.key_conv(x)
        energy =  torch.bmm(query.reshape((query.shape[0],query.shape[2],query.shape[1])), key) #Q.T*K
        
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        out = torch.bmm(proj_value, attention) 
        
        # Add attention weights onto input
        out = self.gamma*out + x
        
        return x


class MainBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        layer=[]
        layer.append(nn.utils.spectral_norm(nn.Conv1d(in_channels = channels, out_channels=channels,kernel_size=1)))
        layer.append(nn.LeakyReLU())

        self.l = nn.Sequential(*layer)
        self.attn = Self_Attn(channels)
        
    
    def forward(self, z):
        out=self.l(z)
        out=self.attn(z)

        return out



class Generator(nn.Module):            

    def _get_noise(time_features):
        time_features = time_features.permute(0, 2, 1)
        bs = time_features.size(0)
        target_len = time_features.size(2)
        noise = torch.randn((bs, 1, target_len))
        noise = torch.cat((time_features, noise), dim=1)
        return noise

    def __init__(self,batch_size, embedding_dim, tau):
        super().__init__()
        self.main=MainBlock(batch_size)
        self.blocks = nn.ModuleList([
           MainBlock(batch_size)
        ])
        self.n=self.blocks.__len__()
        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =2**(self.n+3), out_channels=2**(self.n+3),kernel_size=1))
        self.tau=tau
        self.embedding_dim=embedding_dim
        self.pool=pool(self.tau+self.embedding_dim)
        


    def forward(self, n, phi, X):
        
        n=n[:,None]
        n=n.expand(-1, X.shape[0])
        n=n.reshape(X.shape[0], X.shape[1], X.shape[2])

        aux=X+n
        phi=phi.permute(0, 2, 1).expand(X.size(0), self.embedding_dim, X.size(2))
        
        x = torch.cat((phi, aux), dim=1)

        x=x.reshape((x.shape[0],x.shape[2], x.shape[1]))
        print("x.shape:",x.shape)
        z=self.pool(x)
        print("z.shape:",z.shape)
        
        z=z.reshape((z.shape[0],z.shape[2], z.shape[1]))
        #Main block
        z=self.main(z.reshape((z.shape[1],z.shape[0],1)))

        for b in self.blocks:
            z=nn.functional.interpolate(z.reshape((z.shape[1],1,z.shape[0])),scale_factor=2,mode='linear')
            z=b(z.reshape((z.shape[2],z.shape[0],1)))
        
        z=z.reshape((z.shape[1],z.shape[0],1))

        
        z=self.outlayer(z)
        
        return z



class Discriminator(nn.Module):

    def __init__():
        super().__init__()
    def forward(self, z):
        return z

