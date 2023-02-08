import torch
import torch.nn as nn
import utils
import metrics
import progressive_blocks
import plotting
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam


def train_GAN(
    trainX,
    sequences_train,
    seq_length,
    batch_size,
    discriminator_lr,
    generator_lr,
    num_epochs,
    blocks_to_add,
    timestamp,
    ml,
    fade_in,
    sa,
    save,
    name,
    gpu,
    path,
):
    embedding_dim=10
    value_features=1
    key_features=1

    #extract the number of features
    num_features=trainX.size(2)+10
    
    criterion = nn.MSELoss()

    device=utils.assign_device(gpu)

    #Initializations
    train = TensorDataset(trainX, sequences_train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False) 


    D=progressive_blocks.Discriminator(embedding_dim,seq_length,num_features,batch_size,value_features,key_features,sa,device)
    G=progressive_blocks.Generator(embedding_dim,seq_length,num_features,batch_size,value_features,key_features,sa,device)
    optimD = Adam(D.parameters(), lr=discriminator_lr, betas=(0.9, 0.999))
    optimG = Adam(G.parameters(), lr=generator_lr, betas=(0.9, 0.999))
    #embedder=lstm.LSTMEncoder().to(device)
    #path="Models/M4/"
    embedder=torch.load("Models/Embedder/embedder_model.pt").to(device)

    activeG=(G.step-1)-blocks_to_add
    activeD=blocks_to_add

    utils.create_folder(path+name+'/')

    #Training
    g_losses = []
    d_losses = []
    fids = []
    G.to(device)
    D.to(device)
    fade=1
    sum_fade=0
    g_loss_min=1000000
    d_loss_min=1000000

    print()
    print("Starting training:",name)
    print("Total Epochs: %d \nBlocks to add with fade: %d\nTimestamp to add blocks: %d" % 
                        (num_epochs, blocks_to_add, timestamp))
    print("Fade-in",fade_in)
    print("ML",ml)
    print("SA",sa)
    print()
    for epoch in range(1,num_epochs+1):
            g_losses_temp=[]
            d_losses_temp=[]
            fids_temp=[]
            if (epoch%timestamp==0 and epoch!=0 and activeG!=G.step-1 and activeD!=0 and fade_in==True):
                activeD-=1
                activeG+=1
                fade=0
                sum_fade=1/((timestamp)/2)
                print("Block added")

            elif(fade+sum_fade<=1 and fade_in==True):
                fade+=sum_fade

            else:
                fade=1

            for i, (X, Y) in enumerate((train_loader)):
                X=X.to(device)
                Y=Y.to(device)

                # Generate fake data
                fake_data = G(X,fade,activeG)
                #fake_label = torch.zeros(Y.size(0))
                
            
                # Train the discriminator
                Y=Y[:,:,:fake_data.size(2)]  #we use this to adapt real sequences length to fake sequences length
            
                D.zero_grad()
                d_real_loss = criterion(D(Y,X,fade,activeD), torch.ones_like(D(Y,X,fade,activeD)))
                d_fake_loss = criterion(D(fake_data.detach(),X,fade,activeD), torch.zeros_like(D(fake_data.detach(),X,fade,activeD)))
                d_loss = d_real_loss + d_fake_loss
                d_losses_temp.append(d_loss.item())
                d_loss.backward(retain_graph=False)
                optimD.step()
                
                # Train the generator
                G.zero_grad()
                g_loss = criterion(D(fake_data,X,fade,activeD), torch.ones_like(D(fake_data,X,fade,activeD)))

                if(ml==True):
                    # Add the moment loss
                    g_loss += utils.moment_loss(fake_data, Y)

                g_losses_temp.append(g_loss.item())

                g_loss.backward()
                optimG.step()
                
                #Compute FID
                with torch.no_grad():
                    fake_embedding=embedder(fake_data)
                    real_embedding=embedder(Y) 
                    fid = metrics.calculate_fid(fake_embedding.to("cpu").detach().numpy(),real_embedding.to("cpu").detach().numpy())
                    
                fids_temp.append(fid)    

                # Print the losses
                if (i+1) % 1 == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Fade-in: %f] [FID: %f]" % 
                        (epoch, num_epochs, i+1, len(train_loader), d_loss.item(), g_loss.item(), fade, fid))
                '''
                if(g_loss<g_loss_min and d_loss<d_loss_min and save):
                        g_loss_min = g_loss
                        d_loss_min = d_loss
                        torch.save(G, path+name+'/'+name+'_generator.pt')
                        torch.save(D, path+name+'/'+name+'_discriminator.pt')
                        print('Improvement-Detected, model saved')
                '''

            g_losses.append(torch.mean(torch.Tensor(g_losses_temp)))
            d_losses.append(torch.mean(torch.Tensor(d_losses_temp)))
            fids.append(torch.mean(torch.Tensor(fids_temp)))
            
    values=['Last G loss: '+str(g_losses[-1].item()), 
            'Last D loss: '+str(d_losses[-1].item()),
            'Last FID: '+str(fids[-1].item()),
            'epochs: '+str(num_epochs),
            'ML: '+str(ml),
            'SA: '+str(sa),
            'Fade-in: '+str(fade_in),
            'Blocks to add: '+str(blocks_to_add),
            'Timestamp: '+str(timestamp),
            ]
    torch.save(G, path+name+'/'+name+'_generator.pt')
    torch.save(D, path+name+'/'+name+'_discriminator.pt')
    plotting.plot_training_history('PSA-GAN - M4 - '+name,d_losses, g_losses)
    plotting.plot_fid_history('PSA-GAN - M4 - '+name, fids)
    location=path+'/'+name+'/'+name
    utils.write_file(location, values)

    return D,G, d_losses, g_losses, fids