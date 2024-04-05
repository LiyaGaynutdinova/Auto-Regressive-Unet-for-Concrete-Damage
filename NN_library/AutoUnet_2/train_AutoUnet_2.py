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
    loss_dam = nn.MSELoss(reduction='none')

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
        for i, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['train']):
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            l_seq = 0
            seq = []
            for n in range(10):
                if n == 0:
                    x = torch.cat([geometry, 
                                   (imp_shrinkage[:,[n],:,:]+imp_shrinkage[:,[n+1],:,:])/2, 
                                   damage[:,[n],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, (imp_shrinkage[:,[n],:,:]+imp_shrinkage[:,[n+1],:,:])/2, y.detach()], axis=1)
                # apply the network
                y = net(x)
                l = 0.5*loss_dam(y, (damage[:,[n],:,:]+damage[:,[n+1],:,:])/2).sum()
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y.detach()], axis=1)
                y = net(x)
                # calculate mini-batch losses
                l += loss_dam(y, damage[:,[n+1],:,:]).sum()
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

                if i % 200 == 0:
                    seq.append(y[0,0].detach().cpu())
            
            if i % 200 == 0:
                print(f'Epoch: {epoch} batch: {i} mean train loss: {l_seq/len(x) : 5.10f}')
                save_network(net, args['name'] + f'_{epoch}')
                plot_outputs(damage[0,1:], seq, args['name'] + f'_{epoch}')

        # calculate the loss and accuracy of the validation set
        net.eval()
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        
        L_val = 0

        for j, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['val']):
            seq_val = []
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            for n in range(10):
                if n == 0:
                    x = torch.cat([geometry, 
                                   (imp_shrinkage[:,[n],:,:]+imp_shrinkage[:,[n+1],:,:])/2, 
                                   damage[:,[n],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, (imp_shrinkage[:,[n],:,:]+imp_shrinkage[:,[n+1],:,:])/2, y.detach()], axis=1)
                y = net(x)
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y], axis=1)
                y = net(x)
                L_val += loss_dam(y[:,[0],:,:], damage[:,[n+1],:,:]).sum().detach().cpu()

                if j == 0:
                    seq_val.append(y[0,0].detach().cpu())
            if j == 0:
                plot_outputs(damage[0,1:], seq_val, args['name'] + f'_{epoch}')
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : 5.10f} mean val. rec. loss: {L_val / n_val : 5.10f}')
        save_network(net, args['name'] + f'_{epoch}')        

    return losses_train, losses_val


def plot_outputs_shr(x, fname):
    n_samples = 10
    fig, axs = plt.subplots(nrows=2, ncols=n_samples, figsize=(n_samples, 2), dpi=200)
    for i in range(n_samples):
        axs.flat[i].imshow(x[i].detach().cpu().numpy(), cmap='Greys', vmin=0, vmax=1)
        axs.flat[i].set_axis_off()
    plt.savefig(fname)
    plt.close()

def test(net, loaders, args):
    net.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    loss = nn.MSELoss(reduction='none')

    L_dam = []
    L_stiff = []

    seq_test_dam = []

    for i, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        stiffness = stiffness.to(args['dev'])
        l_seq_dam = []
        l_seq_stiff = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, 
                                   (imp_shrinkage[:,[n],:,:]+imp_shrinkage[:,[n+1],:,:])/2, 
                                   damage[:,[n],:,:]], axis=1)
            else:
                x = torch.cat([geometry, (imp_shrinkage[:,[n],:,:]+imp_shrinkage[:,[n+1],:,:])/2, y.detach()], axis=1)
            y = net(x)
            x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y.detach()], axis=1)
            y = net(x)
            # calculate mini-batch losses
            l_dam = loss(y[:,[0],:,:], damage[:,[n+1],:,:]).sum().detach().cpu().numpy() / (99 * 99)
            l_seq_dam.append(l_dam)
            if i == 0:
                seq_test_dam.append(y[0,0].detach().cpu())
        if i == 0:
            plot_outputs(damage[0,1:], seq_test_dam, args['name'] + f'_test')
        L_dam.append(l_seq_dam)

    return L_dam