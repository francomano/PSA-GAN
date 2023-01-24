import numpy as np
import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class pool(nn.Module):
    def __init__(self, tau):
        #print("tau: ",tau)
        super().__init__()
        str=(math.log2(tau) - 3)
        kernel=int(2**str)
        print("kernel_size: ",kernel)
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

    def __init__(self,batch_size, embedding_dim, tau):
        super().__init__()
        self.main=MainBlock(embedding_dim+2)
        self.blocks = nn.ModuleList([
           MainBlock(embedding_dim+2)
        ])
        self.n=self.blocks.__len__()
        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =embedding_dim+2, out_channels=1,kernel_size=1))
        self.embedding_dim=embedding_dim
        self.pool=pool(tau)
        
    

    def forward(self,phi, X):
        
        #add gaussian noise:
        Xt = X.permute(0, 2, 1)
        bs = Xt.size(0)
        target_len = Xt.size(2)
        noise = torch.randn((bs, 1, target_len))
        noise = torch.cat((Xt, noise), dim=1)
        
        #concatenate with the embedding
        phi=phi.permute(0, 2, 1).expand(Xt.size(0), self.embedding_dim, Xt.size(2))
        x = torch.cat((phi, noise), dim=1)

        #latent space of length 8
        print("x.shape:",x.shape)
        z=self.pool(x)
        print("z.shape:",z.shape)

        z=self.main(z)  #first block(g1)

        ################## Main blocks(g2,gL) #############################

        for b in self.blocks:
            z=nn.functional.interpolate((z),scale_factor=2,mode='linear')
            z=b(z)

        ###########################################################
        
        z=self.outlayer(z)

        z=z.reshape((z.shape[0],z.shape[2]))
        
        return z



class Discriminator(nn.Module):

    def __init__():
        super().__init__()
    def forward(self, z):
        return z

