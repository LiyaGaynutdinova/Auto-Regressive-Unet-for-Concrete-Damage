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
    loss_dam = nn.BCELoss(reduction='none')
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
            seq = []
            for n in range(10):
                if n == 0:
                    x = torch.cat([geometry, 
                                   imp_shrinkage[:,[n],:,:], 
                                   damage[:,[n],:,:]], axis=1)
                    h = torch.zeros(x.shape[0], 6*6*8*net.w, device=args['dev'])
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y], axis=1)
                # apply the network
                y, y_2, h = net(x, h)
                # calculate time-step mini-batch loss
                l_dam = loss_dam(y, damage[:,[n+1],:,:]).sum()
                l_stiff = loss((y_2[:,0]*stiffness[:,0]), stiffness[:,n+1]).sum()
                l_shr = (loss(y_2[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n+1]) / torch.abs(imp_shrinkage[:,n+1,0,0].detach())).sum()
                l = l_dam + l_shr + l_stiff
                # accumulate the sequence loss
                l_seq += l

                if i % 100 == 0:
                    seq.append(y[0,0].detach().cpu())

            # the gradient usually accumulates, need to clear explicitly
            optimizer.zero_grad()
            # compute the gradient from the sequence loss
            l_seq.backward()
            # make the optimization step
            optimizer.step()   
            # accumulate the total loss as a regular float number
            L += l_seq.detach().item()
            
            if i % 100 == 0:
                print(f'Epoch: {epoch} batch: {i} mean train loss: {l_seq/len(x) : 5.10f}')
                save_network(net, args['name'] + f'_{epoch}')
                plot_outputs(damage[0,1:], seq, args['name'] + f'_{epoch}')

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
                if n == 0:
                    x = torch.cat([geometry, 
                                   imp_shrinkage[:,[n],:,:], 
                                   damage[:,[n],:,:]], axis=1)
                    h = torch.zeros(x.shape[0], 6*6*8*net.w, device=args['dev'])
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y], axis=1)
                # apply the network
                y, y_2, h = net(x, h)
                # calculate time-step mini-batch loss
                L_val += loss_dam(y, damage[:,[n+1],:,:]).sum().detach().cpu()
                L_val += loss((y_2[:,0]*stiffness[:,0]), stiffness[:,n+1]).sum().detach().cpu()
                L_val += (loss(y_2[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n+1]) / torch.abs(imp_shrinkage[:,n+1,0,0].detach())).sum().detach().cpu()
                if j == 0:
                    seq_val.append(y[0,0].detach().cpu())
            if j == 0:
                plot_outputs(damage[0,1:], seq_val, args['name'] + f'_{epoch}')
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : 5.10f} mean val. rec. loss: {L_val / n_val : 5.10f}')
        save_network(net, args['name'] + f'_{epoch}')        

    return losses_train, losses_val

def test(net, loaders, args):
    net.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    loss_dam = nn.BCELoss(reduction='none')
    loss = nn.L1Loss(reduction='none')

    L_dam = []
    L_shr = []
    L_stiff = []

    seq_test_dam = []

    for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        obs_shrinkage = obs_shrinkage.to(args['dev'])
        stiffness = stiffness.to(args['dev'])
        l_seq_dam = []
        l_seq_shr = []
        l_seq_stiff = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                h = torch.zeros(x.shape[0], 6*6*8*net.w, device=args['dev'])
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y], axis=1)
            # apply the network
            y, y_2, h = net(x, h)
            # calculate mini-batch losses
            l_dam = loss_dam(y[:,[0],:,:], damage[:,[n+1],:,:]).sum().detach().cpu().numpy()
            l_stiff = loss((y_2[:,0]*stiffness[:,0]), stiffness[:,n+1]).sum().detach().cpu().numpy()
            l_shr = loss(y_2[:,1]*imp_shrinkage[:,-1,0,0].detach(), obs_shrinkage[:,n+1]).sum().detach().cpu().numpy()
            l_seq_dam.append(l_dam)
            l_seq_stiff.append(l_stiff)
            l_seq_shr.append(l_shr)
            if i == 0:
                seq_test_dam.append(y[0,0].detach().cpu())
        if i == 0:
            plot_outputs(damage[0,1:], seq_test_dam, args['name'] + f'_test')
        L_dam.append(l_seq_dam)
        L_stiff.append(l_seq_stiff)
        L_shr.append(l_seq_shr)

    return L_dam, L_stiff, L_shr