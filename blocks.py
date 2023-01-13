import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


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

    def __init__(self,batch_size):
        super().__init__()
        self.main=MainBlock(batch_size)
        self.blocks = nn.ModuleList([
           MainBlock(batch_size)
        ])
        self.n=self.blocks.__len__()
        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =2**(self.n+3), out_channels=2**(self.n+3),kernel_size=1))
        

    def forward(self, z):
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

