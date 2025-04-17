import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from save_load import *


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
            imp_shrinkage = imp_shrinkage.to(args['dev']) / -0.001
            obs_shrinkage = obs_shrinkage.to(args['dev']) / -0.001
            stiffness = stiffness.to(args['dev']) / 30000
            l_seq = 0
            for n in range(11):
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                # apply the network
                y = net(x)
                # calculate mini-batch losses
                l_stiff = loss(y[:,0], stiffness[:,n]).sum()
                l_shr = loss(y[:,1], obs_shrinkage[:,n]).sum()
                l = 1000*l_shr + l_stiff
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
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev']) / -0.001
            obs_shrinkage = obs_shrinkage.to(args['dev']) / -0.001
            stiffness = stiffness.to(args['dev']) / 30000
            for n in range(11):
                x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
                y = net(x).detach()
                l_stiff_val = loss(y[:,0], stiffness[:,n]).sum().detach().cpu()
                l_shr_val = loss(y[:,1], obs_shrinkage[:,n]).sum().detach().cpu()
                L_val += 1000*l_shr_val + l_stiff_val
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : 5.10f} mean val. rec. loss: {L_val / n_val : 5.10f}')
        save_network(net, args['name'] + f'_{epoch}')        

    return losses_train, losses_val


def test_Conv(net, loaders, args):
    net.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    loss = nn.L1Loss(reduction='none')

    L_shr = []
    L_stiff = []

    for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev']) / -0.001
        obs_shrinkage = obs_shrinkage.to(args['dev']) / -0.001
        stiffness = stiffness.to(args['dev']) / 30000
        l_seq_shr = []
        l_seq_stiff = []
        for n in range(1, 11):
            x = torch.cat([geometry, imp_shrinkage[:,[n],:,:], damage[:,[n],:,:]], axis=1)
            y = net(x)
            l_stiff = (loss(y[:,0], stiffness[:,n]) / stiffness[:,n]).sum().detach().cpu()
            l_shr = (loss(y[:,1], obs_shrinkage[:,n]) / obs_shrinkage[:,n]).sum().detach().cpu()
            l_seq_stiff.append(l_stiff)
            l_seq_shr.append(l_shr)
        L_stiff.append(l_seq_stiff)
        L_shr.append(l_seq_shr)

    return L_stiff, L_shr


def test_w_Auto(net, convnet, loaders, args):

    # network
    net.to(args['dev'])
    convnet.to(args['dev'])

    # loss functions
    loss = nn.L1Loss(reduction='none')

    L_shr = []
    L_stiff = []

    # loop fetching a mini-batch of data at each iteration
    for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev']) / -0.001
        obs_shrinkage = obs_shrinkage.to(args['dev']) / -0.001
        stiffness = stiffness.to(args['dev']) / 30000
        l_seq_shr = []
        l_seq_stiff = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[1],:,:], damage[:,[0],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y[:,[0],:,:].detach()], axis=1)
            # apply the network
            y = net(x)
            x_shr = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y[:,[0],:,:].detach()], axis=1)
            y_shr = convnet(x_shr)
            # calculate mini-batch losses
            l_stiff = (loss(y_shr[:,0], stiffness[:,n+1]) / stiffness[:,n+1]).sum().detach().cpu()
            l_shr = (loss(y_shr[:,1], obs_shrinkage[:,n+1]) / obs_shrinkage[:,n+1]).sum().detach().cpu()
            l_seq_stiff.append(l_stiff)
            l_seq_shr.append(l_shr)
        L_stiff.append(l_seq_stiff)
        L_shr.append(l_seq_shr)

    return L_stiff, L_shr


def test_w_Auto_bias(net, convnet, loaders, args):

    # network
    net.to(args['dev'])
    convnet.to(args['dev'])

    # loss functions
    loss = nn.L1Loss(reduction='none')

    L_shr = []
    L_stiff = []

    # loop fetching a mini-batch of data at each iteration
    for i, (geometry, damage, imp_shrinkage, obs_shrinkage, stiffness) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev']) / -0.001
        obs_shrinkage = obs_shrinkage.to(args['dev']) / -0.001
        stiffness = stiffness.to(args['dev']) / 30000
        l_seq_shr = []
        l_seq_stiff = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[1],:,:], damage[:,[0],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y[:,[0],:,:].detach()], axis=1)
            # apply the network
            y = net(x)
            x_shr = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:], y[:,[0],:,:].detach()], axis=1)
            y_shr = convnet(x_shr)
            # calculate mini-batch losses
            l_stiff = ((y_shr[:,0]-stiffness[:,n+1]) / stiffness[:,n+1]).sum().detach().cpu()
            l_shr = ((y_shr[:,1]-obs_shrinkage[:,n+1]) / obs_shrinkage[:,n+1]).sum().detach().cpu()
            l_seq_stiff.append(l_stiff)
            l_seq_shr.append(l_shr)
        L_stiff.append(l_seq_stiff)
        L_shr.append(l_seq_shr)

    return L_stiff, L_shr