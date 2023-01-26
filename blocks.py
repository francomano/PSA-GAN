import numpy as np
import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import get_batch


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

    def __init__(self,embedding_dim, tau,num_features):
        super().__init__()

        self.main=MainBlock(embedding_dim+2)
        self.blocks = nn.ModuleList([MainBlock(embedding_dim+1+num_features)])
        self.num_features=num_features

        #bookkeping of the number of blocks
        self.n=1

        self.tau=tau

        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =embedding_dim+1+num_features+self.n*num_features, out_channels=1,kernel_size=1))
        self.embedding_dim=embedding_dim
        self.pool=pool(tau)
        
    def addBlock(self):
        self.blocks.append(MainBlock(self.embedding_dim+len(self.blocks)+2))
        self.n+=1
        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =self.embedding_dim+1+self.num_features+self.n*self.num_features, out_channels=1,kernel_size=1))
        

    def forward(self,data,batch,epoch):

        #create the time series matrix and the embedded vector
        X=torch.Tensor(get_batch.get_batch(data,self.tau,batch,epoch))
        embedding = nn.Embedding(X.size(0), self.embedding_dim)
        phi = embedding(torch.tensor(np.array(range(X.size(0)))))
        phi=phi[:,None,:]
        
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
        z=self.pool(x)
   

        z=self.main(z)  #first block(g1)

        ################## Main blocks(g2,gL) #############################

        for b in range(0,len(self.blocks)):
            z=nn.functional.interpolate((z),scale_factor=2,mode='linear')
            z=self.blocks[b](z)
            tf=torch.Tensor(get_batch.get_batch(data,2**(3+b+1),batch,epoch)).permute(0, 2, 1) #the X to concatenate
            z=torch.cat((z,tf),dim=1)  #concatenated back to the time features X t:t+τ −1 and forwarded to the next block.

        ###########################################################
        
        z=self.outlayer(z)

        z=z.reshape((z.shape[0],z.shape[2]))
        
        return z



class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, z):
        return z