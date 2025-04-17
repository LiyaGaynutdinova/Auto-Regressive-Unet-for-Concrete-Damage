import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score
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
                                   imp_shrinkage[:,[1],:,:] / -0.001, 
                                   damage[:,[0],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
                # apply the network
                y = net(x)
                # calculate mini-batch losses
                l = loss_dam(y, damage[:,[n+1],:,:]).sum()
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

        for j, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['val']):
            seq_val = []
            geometry = geometry.to(args['dev'])
            damage = damage.to(args['dev'])
            imp_shrinkage = imp_shrinkage.to(args['dev'])
            for n in range(10):
                if n == 0:
                    x = torch.cat([geometry, 
                                   imp_shrinkage[:,[1],:,:] / -0.001, 
                                   damage[:,[0],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
                y = net(x).detach()
                L_val += loss_dam(y, damage[:,[n+1],:,:]).sum().detach().cpu()

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

    loss_damage = nn.L1Loss(reduction='none')
    L_dam = []
    L_dam_total = []
    seq_test_dam = []
    seq_test_dam_total = []

    for i, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        l_seq_dam = []
        l_seq_dam_total = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, damage[:,[0],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
            # apply the network
            y = net(x)
            # calculate mini-batch losses
            l_dam = (loss_damage(y, damage[:,[n+1],:,:])[torch.where(geometry>0)]).mean().detach().cpu().numpy()
            l_dam_total = (torch.abs(y.sum()-damage[:,[n+1],:,:].sum())/damage[:,[n+1],:,:].sum()).detach().cpu().numpy()
            l_seq_dam.append(l_dam)
            l_seq_dam_total.append(l_dam_total)
            if i == 0:
                seq_test_dam.append(y[0,0].detach().cpu())
        if i == 0:
            plot_outputs(damage[0,1:], seq_test_dam, args['name'] + f'_test')
        L_dam.append(l_seq_dam)
        L_dam_total.append(l_seq_dam_total)

    return L_dam, L_dam_total


def test_bias(net, loaders, args):
    net.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    L_dam_total = []
    seq_test_dam_total = []

    for i, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        l_seq_dam_total = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, damage[:,[0],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
            # apply the network
            y = net(x)
            # calculate mini-batch losses
            l_dam_total = ((y.sum()-damage[:,[n+1],:,:].sum())/damage[:,[n+1],:,:].sum()).detach().cpu().numpy()
            l_seq_dam_total.append(l_dam_total)
        L_dam_total.append(l_seq_dam_total)

    return L_dam_total


def test_F1(net, loaders, args):
    net.to(args['dev'])

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    L_dam_total = []

    for i, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        l_seq_dam_total = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, damage[:,[0],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
            # apply the network
            y = net(x)
            # calculate mini-batch losses
            y_bin = torch.where(y>0.25,1.,0.).flatten()
            dam_bin = torch.where(damage[:,n+1,:,:]>0.5,1.,0.).flatten()
            l_dam_total = multiclass_f1_score(y_bin, dam_bin, num_classes=2).detach().cpu().numpy()
            l_seq_dam_total.append(l_dam_total)
        L_dam_total.append(l_seq_dam_total)

    return L_dam_total


def test_blur(net, loaders, args):
    net.to(args['dev'])

    weight = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float, device=args['dev']).view(1,1,3,3)

    if args['dev'] == "cuda":
        torch.cuda.empty_cache() 

    Var = []
    Var_true = []

    for i, (geometry, damage, imp_shrinkage, _, _) in enumerate(loaders['test']):
        geometry = geometry.to(args['dev'])
        damage = damage.to(args['dev'])
        imp_shrinkage = imp_shrinkage.to(args['dev'])
        Var_true_seq = []
        Var_seq = []
        for n in range(10):
            if n == 0:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, damage[:,[0],:,:]], axis=1)
            else:
                x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
            # apply the network
            y = net(x)
            # calculate mini-batch losses
            im_grad = nn.functional.conv2d(y, weight)
            im_grad_true = nn.functional.conv2d(damage[:,[n+1],:,:], weight)
            Var_seq.append(im_grad.var().detach().cpu().numpy())
            Var_true_seq.append(im_grad_true.var().detach().cpu().numpy())
        Var.append(Var_seq)
        Var_true.append(Var_true_seq)

    return Var, Var_true


def train_w_Conv(net, convnet, loaders, args):

    # network
    net.to(args['dev'])
    convnet.to(args['dev'])

    # loss functions
    loss_dam = nn.MSELoss(reduction='none')
    loss = nn.L1Loss(reduction='none')

    # optimizer
    params = list(net.parameters()) + list(convnet.parameters())
    optimizer = optim.Adam(params, lr = args['lr'])
    
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
                                   imp_shrinkage[:,[1],:,:] / -0.001, 
                                   damage[:,[0],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
                # apply the network
                y = net(x)
                x_shr = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y], axis=1)
                y_shr = convnet(x_shr)
                # calculate mini-batch losses
                l_dam = loss_dam(y, damage[:,[n+1],:,:]).sum() / (99 * 99)
                l_stiff = (loss(y_shr[:,0]*stiffness[:,0].detach(), stiffness[:,n+1]) / stiffness[:,n+1].detach()).sum()
                l_shr = (loss(y_shr[:,1]*(-0.001), obs_shrinkage[:,n+1]) / torch.abs(obs_shrinkage[:,n+1]).detach()).sum()
                l = l_dam + l_stiff + l_shr
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
                save_network(convnet, args['conv_name'] + f'_{epoch}')
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
                    x = torch.cat([geometry, imp_shrinkage[:,[1],:,:] / -0.001, damage[:,[0],:,:]], axis=1)
                else:
                    x = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y.detach()], axis=1)
                y = net(x)
                x_shr = torch.cat([geometry, imp_shrinkage[:,[n+1],:,:] / -0.001, y], axis=1)
                y_shr = convnet(x_shr)
                l_dam = loss_dam(y, damage[:,[n+1],:,:]).sum() / (99 * 99)
                l_stiff = (loss(y_shr[:,0]*stiffness[:,0].detach(), stiffness[:,n+1]) / stiffness[:,0].detach()).sum()
                l_shr = (loss(y_shr[:,1]*(-0.001), obs_shrinkage[:,n+1]) / torch.abs(obs_shrinkage[:,n+1])).sum()
                L_val += (l_dam.detach() + l_stiff.detach() + l_shr.detach()).cpu()

                if j == 0:
                    seq_val.append(y[0,0].detach().cpu())
            if j == 0:
                plot_outputs(damage[0,1:], seq_val, args['name'] + f'_{epoch}')
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : 5.10f} mean val. rec. loss: {L_val / n_val : 5.10f}')
        save_network(net, args['name'] + f'_{epoch}')  
        save_network(convnet, args['conv_name'] + f'_{epoch}')      

    return losses_train, losses_val