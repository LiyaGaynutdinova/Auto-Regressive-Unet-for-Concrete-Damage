import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from save_load import *

def plot_outputs(x, y, fname):
    n_samples = 10
    fig, axs = plt.subplots(nrows=2, ncols=n_samples, figsize=(n_samples, 2), dpi=200)
    for i in range(n_samples):
        axs.flat[i].imshow(x[i].detach().cpu().numpy(), cmap='Greys', vmin=0, vmax=1)
        axs.flat[i].set_axis_off()
    for i in range(n_samples, 2*n_samples):
        axs.flat[i].imshow(y[i-n_samples].numpy(), cmap='Greys', vmin=0, vmax=1)
        axs.flat[i].set_axis_off()
    plt.savefig(fname)
    plt.close()


def train(net, loaders, args):

    # network
    net.to(args['dev'])

    # loss functions
    loss = nn.L1Loss(reduction='none')

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])

    n_train = len(loaders['train'].dataset)
    n_val = len(loaders['val'].dataset)

    losses_train = []
    losses_val = []

    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        # accumulate total loss over the dataset
        L = 0
        net.train()
        # loop fetching a mini-batch of data at each iteration
        for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['train']):
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            obs_shrinkage = obs_shrinkage.to(args['dev'])
            stiffness = stiffness.to(args['dev'])
            l_seq = 0
            for n in range(11):
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                # apply the network
                y = net(x)
                # calculate mini-batch losses
                l_stiff = (loss(y[:,0]*stiffness[:,0].detach(), stiffness[:,n]) / stiffness[:,0].detach()).sum()
                l_shr = (loss(y[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n]) / torch.abs(imp_shrinkage[:,-1,0,0].detach())).sum()
                l = l_shr + l_stiff
                # accumulate the total loss as a regular float number
                loss_batch = l.detach().item()
                L += loss_batch
                l_seq += loss_batch
                # the gradient usually accumulates, need to clear explicitly
                optimizer.zero_grad()
                # compute the gradient from the mini-batch loss
                l.backward()
                # make the optimization step
                optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch: {epoch} batch: {i} mean train loss: {l_seq/len(x) : 5.10f}')
                save_network(net, args['name'] + f'_{epoch}')

        # calculate the loss and accuracy of the validation set
        net.eval()
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        
        L_val = 0

        for j, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['val']):
            seq_val = []
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            obs_shrinkage = obs_shrinkage.to(args['dev'])
            stiffness = stiffness.to(args['dev'])
            for n in range(10):
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                y = net(x).detach()
                l_stiff_val = (loss(y[:,0]*stiffness[:,0].detach(), stiffness[:,n]) / stiffness[:,0].detach()).sum().detach().cpu()
                l_shr_val = (loss(y[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n]) / torch.abs(imp_shrinkage[:,-1,0,0].detach())).sum().detach().cpu()
                L_val += l_shr_val + l_stiff_val
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : 5.10f} mean val. rec. loss: {L_val / n_val : 5.10f}')
        save_network(net, args['name'] + f'_{epoch}')        

    return losses_train, losses_val

def test(net, loaders, args):
    net.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    loss = nn.L1Loss(reduction='none')

    L_shr = []
    L_stiff = []

    for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        obs_shrinkage = obs_shrinkage.to(args['dev'])
        stiffness = stiffness.to(args['dev'])
        l_seq_shr = []
        l_seq_stiff = []
        for n in range(11):
            x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
            y = net(x)
            l_stiff = (loss(y[:,0]*stiffness[:,0].detach(), stiffness[:,n]) / stiffness[:,0].detach()).sum().detach().cpu()
            l_shr = (loss(y[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n]) / torch.abs(imp_shrinkage[:,-1,0,0].detach())).sum().detach().cpu()
            l_seq_stiff.append(l_stiff)
            l_seq_shr.append(l_shr)
        L_stiff.append(l_seq_stiff)
        L_shr.append(l_seq_shr)

    return L_stiff, L_shr


def test_w_Autonet(net, unet, loaders, args):
    net.to(args['dev'])
    unet.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    loss = nn.L1Loss(reduction='none')

    L_shr = []
    L_stiff = []

    for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        obs_shrinkage = obs_shrinkage.to(args['dev'])
        stiffness = stiffness.to(args['dev'])
        l_seq_shr = []
        l_seq_stiff = []
        for n in range(10):
            if n == 0:
                x_unet = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
            else:
                x_unet = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y_unet[:,[0],:,:].detach()], axis=1)
            y_unet = unet(x_unet)
            x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y_unet[:,[0],:,:].detach()], axis=1)
            y = net(x)
            l_stiff = (loss(y[:,0]*stiffness[:,0].detach(), stiffness[:,n]) / stiffness[:,0].detach()).sum().detach().cpu()
            l_shr = (loss(y[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n]) / torch.abs(imp_shrinkage[:,-1,0,0].detach())).sum().detach().cpu()
            l_seq_stiff.append(l_stiff)
            l_seq_shr.append(l_shr)
        L_stiff.append(l_seq_stiff)
        L_shr.append(l_seq_shr)

    return L_stiff, L_shr