import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from save_load import *


def plot_outputs(x, y, fname):
    n_samples = len(x)
    fig, axs = plt.subplots(nrows=2, ncols=n_samples, figsize=(n_samples, 2), dpi=100)
    for i in range(n_samples):
        axs.flat[i].imshow(x[i][0].detach().cpu().numpy(), cmap='Greys')
        axs.flat[i].set_axis_off()
    for i in range(n_samples, 2*n_samples):
        axs.flat[i].imshow(y[i-n_samples][0].detach().cpu().numpy(), cmap='Greys')
        axs.flat[i].set_axis_off()
    plt.savefig(fname)
    plt.close()


def train(net, loaders, args):

    # network
    net.to(args['dev'])

    # loss function
    loss = nn.BCELoss(reduction='none')

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
        for i, (x, label) in enumerate(loaders['train']):
            x = x.to(args['dev'])
            # apply the network
            y = net(x)
            y_filtered = y * x.round()
            # calculate mini-batch losses
            l = loss(y_filtered, label).sum()
            # accumulate the total loss as a regular float number
            loss_batch = l.detach().item()
            L += loss_batch
            # the gradient usually accumulates, need to clear explicitly
            optimizer.zero_grad()
            # compute the gradient from the mini-batch loss
            l.mean().backward()
            # make the optimization step
            optimizer.step()
            
            if i % 50 == 0:
                print(f'Epoch: {epoch} batch: {i} mean train loss: {loss_batch/len(x) : 5.10f}')
                save_network(net, args['name'] + f'_{epoch}')

        # calculate the loss and accuracy of the validation set
        net.eval()
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        
        L_val = 0
        for j, (x_val, label_val) in enumerate(loaders['val']):
            x_val = x_val.to(args['dev'])
            y_val = net(x_val)
            y_val_filtered = y * x.round()
            L_val += loss(y_val_filtered, label_val).detach().sum().item()
        losses_train.append(L / n_train)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : 5.10f} mean val. rec. loss: {L_val / n_val : 5.10f}')
        if epoch % 5 == 0:
            save_network(net, args['name'] + f'_{epoch}')
            plot_outputs(x_val, y_val, args['name'] + f'_{epoch}')
    save_network(net, args['name'])
    plot_outputs(label_val, y_val, args['name'] + f'_{epoch}')

    return losses_train, losses_val