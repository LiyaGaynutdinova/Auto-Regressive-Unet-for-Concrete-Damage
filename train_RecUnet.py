import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from save_load import *

def plot_outputs(x, y, fname):
    n_samples = 10
    fig, axs = plt.subplots(nrows=2, ncols=n_samples, figsize=(n_samples, 2), dpi=200)
    for i in range(n_samples):
        axs.flat[i].imshow(x[i].detach().cpu().numpy(), cmap='Greys')
        axs.flat[i].set_axis_off()
    for i in range(n_samples, 2*n_samples):
        axs.flat[i].imshow(y[i-n_samples].numpy(), cmap='Greys')
        axs.flat[i].set_axis_off()
    plt.savefig(fname)
    plt.close()


def train(net, loaders, args):

    # network
    net.to(args['dev'])

    # loss functions
    loss_damage = nn.BCELoss(reduction='none')
    loss_shrinkage = nn.L1Loss(reduction='none')

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
        for i, (geometry, damage, imp_shrinkage, obs_shrinkage, _) in enumerate(loaders['train']):
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            obs_shrinkage = obs_shrinkage.to(args['dev'])
            l_seq = 0
            seq = []
            for n in range(10):
                if n == 0:
                    x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y.detach()], axis=1)
                # apply the network
                y = net(x)
                # calculate mini-batch losses
                l_dam = loss_damage(y[:,[0],:,:], damage[:,[n+1],:,:]).sum()
                l_shr = loss_shrinkage((y[:,[1],:,:]*imp_shrinkage[:,[n],:,:].detach()).mean((1,2,3)), obs_shrinkage[:,n+1]).sum()
                l = l_dam + l_shr
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
                    seq.append(y[0,0].detach().cpu())
            
            if i % 100 == 0:
                print(f'Epoch: {epoch} batch: {i} mean train loss: {l_seq/len(x) : 5.10f}')
                save_network(net, args['name'] + f'_{epoch}')
                plot_outputs(damage[0,1:], seq, args['name'] + f'_{epoch}')

        # calculate the loss and accuracy of the validation set
        net.eval()
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        
        L_val = 0
        seq_val = []

        for j, (geometry, damage, imp_shrinkage, obs_shrinkage, _) in enumerate(loaders['val']):
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            obs_shrinkage = obs_shrinkage.to(args['dev'])
            for n in range(10):
                if n == 0:
                    x_val = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                else:
                    x_val = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y_val.detach()], axis=1)
                y_val = net(x_val)
                l_dam_val = loss_damage(y_val[:,[0],:,:], damage[:,[n+1],:,:]).sum().detach().item()
                l_shr_val = loss_shrinkage((y_val[:,[1],:,:]*imp_shrinkage[:,[n],:,:].detach()).mean((1,2,3)), obs_shrinkage[:,n+1]).sum().detach().item()
                L_val += l_dam_val + l_shr_val

                if j == 0:
                    seq_val.append(y_val[0,0].detach().cpu())
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

    loss_damage = nn.BCELoss(reduction='none')
    loss_shrinkage = nn.L1Loss(reduction='none')

    L_dam = []
    L_shr = []

    seq_test_dam = []
    seq_test_shr = []

    for i, (geometry, damage, imp_shrinkage, obs_shrinkage) in enumerate(loaders['test']):
        print(i)
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        obs_shrinkage = obs_shrinkage.to(args['dev'])
        l_seq_dam = []
        l_seq_shr = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], y.detach()], axis=1)
            # apply the network
            y = net(x)
            # calculate mini-batch losses
            l_dam = loss_damage(y[:,[0],:,:], damage[:,[n+1],:,:]).sum().detach().cpu().numpy()         
            l_shr = loss_shrinkage(y[:,[1],:,].mean((1,2,3))*imp_shrinkage[:,[n],:,:].detach(), obs_shrinkage[:,n+1]).sum().detach().cpu().numpy()
            if n>0:
                #l_dam /= damage[:,[n+1],:,:].sum().detach().cpu().numpy()
                l_shr /= obs_shrinkage[:,n+1].abs().sum().detach().cpu().numpy()
            l_seq_dam.append(l_dam)
            l_seq_shr.append(l_shr)
            if i == 0:
                seq_test_dam.append(y[0,0].detach().cpu())
                seq_test_shr.append(y[0,1].detach().cpu())
        if i == 0:
            plot_outputs(seq_test_dam, seq_test_shr, args['name'] + f'_test_shrinkage')
        L_dam.append(l_seq_dam)
        L_shr.append(l_seq_shr)

    return L_dam, L_shr