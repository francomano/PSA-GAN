import numpy as np
import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import utils


class pool(nn.Module):
    def __init__(self, tau,step):
        #print("tau: ",tau)
        super().__init__()
        str=(step - 1)
        kernel=int(2**str)
        self.pool=nn.AvgPool1d(kernel_size=kernel)
    def forward(self,x):
        z=self.pool(x)
        return z



class Self_Attn(nn.Module):
   
    def __init__(self, in_dim,value_features,key_features):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = key_features , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = key_features , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = value_features , kernel_size= 1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
        
    def forward(self,x):
        
        
        query  = self.query_conv(x)
        key =  self.key_conv(x)
        energy =  torch.bmm(query.permute(0,2,1), key) #Q.T*K
        
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        out = torch.bmm(attention,proj_value.permute(0,2,1)).permute(0,2,1) 
        
        # Add attention weights onto input
        out = self.gamma*out + x
        
        return out


class MainBlock(nn.Module):
    def __init__(self,inc,outc,value_features,key_features):
        super().__init__()
        layer=[]
        layer.append(nn.utils.spectral_norm(nn.Conv1d(in_channels = inc, out_channels=outc,kernel_size=1)))
        layer.append(nn.LeakyReLU())

        self.l = nn.Sequential(*layer)
        self.attn = Self_Attn(outc,value_features,key_features)
        
    
    def forward(self, z):
        out=self.l(z)
        out=self.attn(out)
        return out



class Generator(nn.Module):            

    def __init__(self,embedding_dim,fake_len,num_features,batch_size,value_features,key_features):
        super().__init__()

        self.main=MainBlock(1+num_features,32,value_features,key_features)

        self.fake_len=fake_len  #the len we want to reach
        self.step=(int)(math.log2(self.fake_len))-2
        self.embedding_dim=embedding_dim
        self.batch_size=batch_size


        self.blocks = nn.ModuleList([])
        for i in range(self.step-1):
            self.blocks.append(MainBlock(32+num_features,32,value_features,key_features))

        

        

        self.embedding=nn.Embedding(batch_size,embedding_dim)
        self.pool=pool(self.fake_len,self.step)
        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =32, out_channels=1,kernel_size=1))
        
        
    

    def forward(self, X):
        
        Xt = X.permute(0, 2, 1)

        #add gaussian noise:
        Xt=utils.noise(Xt)
        
        #concatenate with the embedding
        phi=self.embedding(torch.tensor(np.array(range(self.batch_size))))
        phi=phi.unsqueeze(1)
        phi=phi.permute(0, 2, 1)
        phi=phi.expand(Xt.size(0), self.embedding_dim, Xt.size(2))
        
        x = torch.cat((phi, Xt), dim=1)

        #latent space of length 8
        z=self.pool(x)

        #first block(g1)
        z=self.main(z)  

        ################## Main blocks(g2,gL) #############################

        
        for idx,b in enumerate(self.blocks[:self.step-1]):
            z=nn.functional.interpolate((z),scale_factor=2,mode='linear')
            tf = F.avg_pool1d(x[:, :-1, :], kernel_size=2 ** (self.step - 1 - (idx + 1))) 
            z=torch.cat((z,tf),dim=1)  #concatenated back to the time features X t:t+τ −1 and forwarded to the next block.
            z=b(z)

        ###########################################################
        
        z=self.outlayer(z).squeeze(1)
        z=z.unsqueeze(dim=1)
        
        return z



class Discriminator(nn.Module):

    def __init__(self,embedding_dim,fake_len,num_features,batch_size,value_features,key_features):
        super().__init__()

        self.fake_len=fake_len
        self.step=(int)(math.log2(fake_len))-2   #number of blocks we used to reach fake_len
        self.embedding_dim=embedding_dim
        self.batch_size=batch_size

        self.embedding=nn.Embedding(batch_size,embedding_dim)
        first_module=[]
        first_module.append(nn.utils.spectral_norm(nn.Conv1d(in_channels = num_features+1, out_channels=32,kernel_size=1)))
        first_module.append(nn.LeakyReLU())
        self.first_module = nn.Sequential(*first_module)

        self.blocks = nn.ModuleList([])
        n=self.step
        while(n-1): #1 is the first main block
            self.blocks.append(MainBlock(32,32,value_features,key_features))
            n-=1

        last_module=[]
        last_module.append(MainBlock(32,32,value_features,key_features))
        last_module.append(nn.utils.spectral_norm(nn.Conv1d(in_channels =32, out_channels=1,kernel_size=1)))
        last_module.append(nn.LeakyReLU())

        self.last_module = nn.Sequential(*last_module)

        self.fc = nn.utils.spectral_norm(nn.Linear(8, 1))




    def forward(self,Z,X):    #Z is the output of the generator (batch,1,fake_len), X the time-features matrix

        #compute how many blocks we should use to arrive at the original length from 8(latent space)
        reduce_factor = int(math.log2(self.fake_len)) - int(math.log2(Z.size(2)))
        X=X.permute(0,2,1)

        #embedding
        phi=self.embedding(torch.tensor(np.array(range(self.batch_size))))
        phi=phi.unsqueeze(1)
        phi=phi.permute(0, 2, 1)
        phi=phi.expand(X.size(0), self.embedding_dim, X.size(2))
        X = torch.cat((phi, X), dim=1)

        reduced_X = F.avg_pool1d(X, kernel_size=2 ** reduce_factor) #in order to concatenate with Z
        x = torch.cat((reduced_X, Z), dim=1)

        #D->32 channels
        x = self.first_module(x)
        

        for l in self.blocks:
            x = l(x)
            x = F.avg_pool1d(x, kernel_size=2)
            
            

        x = self.last_module(x)
        
        x=x.squeeze(dim=1)
        x = self.fc(x)
        
        return x