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
    
        
    def __init__(self,embedding_dim,fake_len,num_features,batch_size,value_features,key_features,device):
        super().__init__()

        self.main=MainBlock(1+num_features,32,value_features,key_features)

        self.fake_len=fake_len  #the len we want to reach
        self.step=(int)(math.log2(self.fake_len))-2
        self.embedding_dim=embedding_dim
        self.batch_size=batch_size
        self.device=device
        self.softmax = nn.Softmax(dim=1)
        
        self.blocks = nn.ModuleList([])
        for i in range(self.step-1):
            self.blocks.append(MainBlock(32+num_features,32,value_features,key_features))
        
        self.skip_block = nn.ModuleList([])
        for i in range(1, self.step-1):
            self.skip_block.append(nn.Conv1d(in_channels=32,out_channels=32,kernel_size=1))


        self.embedding=nn.Embedding(batch_size,embedding_dim)
        self.pool=pool(self.fake_len,self.step)
        self.outlayer=nn.utils.spectral_norm(nn.Conv1d(in_channels =32, out_channels=1,kernel_size=1))
        

    def softmax_min_max_localscaling(self, target, alpha=100):
        min = torch.sum(target * self.softmax(-alpha * target), dim=1, keepdim=True)
        max = torch.sum(target * self.softmax(alpha * target), dim=1, keepdim=True)

        return (target - min) / (max - min)    
    
    


    def forward(self, X, fade, active):
        
        #Parameters
        if(active == None):
            active = self.step-1
            
        Xt = X.permute(0, 2, 1)
        
        #add gaussian noise:
        Xt=utils.noise(Xt, self.device)

        #concatenate with the embedding
        phi=self.embedding(torch.tensor(np.array(range(self.batch_size))).to(self.device))
        phi=phi.unsqueeze(1)
        phi=phi.permute(0, 2, 1)
        phi=phi.expand(Xt.size(0), self.embedding_dim, Xt.size(2))
        x = torch.cat((phi, Xt), dim=1)

        #latent space of length 8
        z=self.pool(x)

        #first block(g1)
        z=self.main(z)  

        ################## Main blocks(g2,gL) #############################

        
        for i,b in enumerate(self.blocks[:active]):
            z=nn.functional.interpolate((z),scale_factor=2,mode='linear')
            temp_z=z
            tf = F.avg_pool1d(x[:, :-1, :], kernel_size=2 ** (self.step - 1 - (i + 1))) 
            z=torch.cat((z,tf),dim=1)  #concatenated back to the time features X t:t+τ −1 and forwarded to the next block.
            z=b(z)
            temp_i=i-1

        ###########################################################
        
        if (fade and active > 0):
           layers= self.skip_block[temp_i]
           z=fade*self.outlayer(z).squeeze(1)+(1-fade)*(self.outlayer(layers(temp_z)).squeeze(1))
           #z=self.outlayer(z).squeeze(1)
        else:
           z=self.outlayer(z).squeeze(1)


        scale=False
        if (scale==True):
            z=self.softmax_min_max_localscaling(z)


        z=z.unsqueeze(dim=1)

        
        
        return z



class Discriminator(nn.Module):

    def __init__(self,embedding_dim,fake_len,num_features,batch_size,value_features,key_features,device):
        super().__init__()

        self.fake_len=fake_len
        self.step=(int)(math.log2(fake_len))-2   #number of blocks we used to reach fake_len
        self.embedding_dim=embedding_dim
        self.batch_size=batch_size
        self.device=device

        self.embedding=nn.Embedding(batch_size,embedding_dim)
        first_module=[]
        first_module.append(nn.utils.spectral_norm(nn.Conv1d(in_channels = num_features+1, out_channels=32,kernel_size=1)))
        first_module.append(nn.LeakyReLU())
        self.first_module = nn.Sequential(*first_module)

        self.blocks = nn.ModuleList([])
        n=self.step-1
        while(n>0): 
            self.blocks.append(MainBlock(32,32,value_features,key_features))
            n-=1
        
        last_module=[]
        last_module.append(MainBlock(32,32,value_features,key_features))
        last_module.append(nn.utils.spectral_norm(nn.Conv1d(in_channels =32, out_channels=1,kernel_size=1)))
        last_module.append(nn.LeakyReLU())

        self.last_module = nn.Sequential(*last_module)

        self.fc = nn.utils.spectral_norm(nn.Linear(8, 1))




    def forward(self,Z,X,fade,active):    #Z is the output of the generator (batch,1,fake_len), X the time-features matrix

        reduce_factor = int(math.log2(self.fake_len)) - int(math.log2(Z.size(2)))
        X=X.permute(0,2,1)

        #embedding
        phi=self.embedding(torch.tensor(np.array(range(self.batch_size))).to(self.device))
        phi=phi.unsqueeze(1)
        phi=phi.permute(0, 2, 1)
        phi=phi.expand(X.size(0), self.embedding_dim, X.size(2))
        X = torch.cat((phi, X), dim=1)

        reduced_X = F.avg_pool1d(X, kernel_size=2 ** reduce_factor) #in order to concatenate with Z
       

        x = torch.cat((reduced_X, Z), dim=1)

        #D->32 channels
        x = self.first_module(x)
        
        
        for i,l in enumerate(self.blocks[active:]):
            if(i==0):
                x=fade*l(x)+(1-fade)*l(x)
                x = F.avg_pool1d(x, kernel_size=2)
            else:    
                x = l(x)
                x = F.avg_pool1d(x, kernel_size=2)
        
        

        x = self.last_module(x)
        
        x=x.squeeze(dim=1)
        x = self.fc(x)
        
        return x