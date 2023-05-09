'''
This file defines the models and the MNIST dataset before creating functionalities to train. The majority of this file is from the public
codebase at https://github.com/mike-fang/imprecise_optical_neural_network. The function mnist_complex has been significantly modified by me.

@version 3.8.2021
'''

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from time import time
from torchvision import datasets, transforms
import matplotlib.pylab as plt
from optical_nn import *
import complex_torch_var as ct
from time import time
import os
from default_params import *
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import math


TEST_SIZE = 32
BATCH_SIZE = 32
USE_CUDA = True
if USE_CUDA:
    print('Using CUDA')
    DEVICE = th.device('cuda')
else:
    print('Using CPU')
    DEVICE = th.device('cpu')


# Define loader
def mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=True):
    loader =  th.utils.data.DataLoader(datasets.ImageFolder('Data/catdog/train', 
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.Resize((64,64)),
                    transforms.ToTensor()])),
            batch_size=batch_size, shuffle=shuffle)
    return loader

def mnist_val_loader(train=False, batch_size=BATCH_SIZE, shuffle=True):
    loader =  th.utils.data.DataLoader(datasets.ImageFolder('Data/catdog/val', 
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.Resize((64,64)),
                    transforms.ToTensor()])),
            batch_size=batch_size, shuffle=shuffle)
    return loader


def mnist_test_loader(train=False, batch_size=BATCH_SIZE, shuffle=True):
    loader =  th.utils.data.DataLoader(datasets.ImageFolder('Data/catdog/test', 
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.Resize((64,64)),
                    transforms.ToTensor()])),
            batch_size=batch_size, shuffle=shuffle)
    return loader

# Get X for testing
for data, target in mnist_test_loader(train=False, batch_size=BATCH_SIZE, shuffle=False):
    continue
data = data.view(-1, 64**2)
data, target = data.to(DEVICE), target.to(DEVICE)
print(data.shape)
X0 = data[7][None, :]


# Network dims
N_IN = 64**2//2


def train(model, n_epochs, log_interval, optim_params, batch_size=BATCH_SIZE, criterion=nn.NLLLoss(), device=DEVICE, epoch_callback=None, log_callback=None, writer=None, iteration=0):
    loader = mnist_loader(train=True, batch_size=batch_size)
    validloader = mnist_val_loader(batch_size=batch_size)
    optimizer = optim.SGD(model.parameters(), **optim_params)
    #criterion = nn.CrossEntropyLoss()
    valid_loss_min = np.inf
    tx = time()
    t0 = time()
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data = data.view(-1, 64**2)
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1/ (batch_idx + 1 ))*(loss.data-train_loss))
            if batch_idx % log_interval == 0 and batch_idx != 0:
                t = time() - t0
                t0 = time()
                acc = (out.argmax(1) == target).float().mean()
                #acc = get_acc(model, device)
                out = model(data)
                print(f'Epoch: {epoch}, Batch: {batch_idx:.4f}/{ceil((n_epochs*len(loader))):.1f}, Train loss: {loss.float():.4f}, Train acc: {acc:.4f}, Time/it: {t/log_interval * 1e3:.4f} ms')
                if not (writer is None):
                    print(n_epochs*iteration*len(loader)/batch_size + epoch*len(loader)/batch_size + batch_idx)
                    writer.add_scalar('training loss', loss.float(), n_epochs*iteration*len(loader)/batch_size + epoch*len(loader)/batch_size + batch_idx)
                    writer.add_scalar('training accuracy', acc, n_epochs*iteration*len(loader)/batch_size + epoch*len(loader)/batch_size + batch_idx)
                if log_callback:
                    log_callback(model, epoch)
        # Model validation
        model.eval()
        for batch_idx, (data,target) in enumerate(validloader):
            # Move to GPU
            data = data.view(-1, 64**2)
            data, target = data.to(device), target.to(device)
            # Update the average validation loss
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the batch loss
            loss = criterion(output, target)
            # Update the average validation loss
            valid_loss = valid_loss + ((1/ (batch_idx +1)) * (loss.data - valid_loss))
    
        # print training/validation stats
        print('Epoch: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
            epoch,
            train_loss,
            valid_loss))
    
        # Save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            #torch.save(model.state_dict(), os.path.join('model','model_transfer.pt'))
            valid_loss_min = valid_loss
        if epoch_callback:
            epoch_callback(model, epoch)
        print("EpochTime:" + str(time() - tx))
        tx = time()


def get_acc(model, device=DEVICE):
    confusion = None
    with th.no_grad():
        for data, target in mnist_test_loader(train=False, batch_size=TEST_SIZE):
            data = data.view(-1, 64**2)
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(1)
            acc = (pred == target).float().mean()
            confusion = confusion_matrix(target.cpu(), pred.cpu())
    return acc.item(), confusion


def test(model, criterion= nn.NLLLoss(), device=DEVICE):
        
        loader = mnist_test_loader(batch_size=TEST_SIZE)
        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        model.eval() #set model into evaluation/testing mode. It turns of drop off layer
        #Iterating over test data
        for batch_idx, (data, target) in enumerate(loader):
            # move to GPU
            data = data.view(-1, 64**2)
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to 
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
                
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))


class StackedFFTUnitary(nn.Sequential):
    def __init__(self, D, n_stack=None, sigma_PS=0, sigma_BS=0):
        if n_stack is None:
            P = int(log2(D))
            n_stack = int(D//P)
        layers = [FFTUnitary(D, sigma_PS=sigma_PS, sigma_BS=sigma_BS) for _ in range(n_stack)]

        super().__init__(*layers)
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
    @property
    def sigma_PS(self):
        return self._sigma_PS
    @property
    def sigma_BS(self):
        return self._sigma_BS
    @sigma_PS.setter
    def sigma_PS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_PS = new_sig
        self._sigma_PS = new_sig
    @sigma_BS.setter
    def sigma_BS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_BS = new_sig
        self._sigma_BS = new_sig

def mnist_stacked_fft(n_stack, device=DEVICE, T0=0.03):
    f = ShiftedSoftplus(T=0.03)
    layers = [
            Linear(N_IN, 256),
            ModNonlinearity(f=f),
            StackedFFTUnitary(256, n_stack=n_stack),
            Diagonal(256, 256),
            StackedFFTUnitary(256, n_stack=n_stack),
            ModNonlinearity(f=f),
            Linear(256, 10),
            ComplexNorm(),
            nn.LogSoftmax(dim=1)
            ]
    net = NoisySequential(*layers).to(device)
    return net

def mnist_grid_truncated(num_in=N_IN, num_out=10, hidden_units=[256, 256], device=DEVICE, sigma_PS=0, sigma_BS=0, T0=0.03):
    f = ShiftedSoftplus(T=T0)
    layers = [
        TruncatedGridLinear(num_in, hidden_units[0], sigma_PS=sigma_PS, sigma_BS=sigma_BS),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            TruncatedGridLinear(nh_, nh, sigma_PS=sigma_PS, sigma_BS=sigma_BS),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        TruncatedGridLinear(hidden_units[-1], num_out, sigma_PS=sigma_PS, sigma_BS=sigma_BS),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ])
    net = NoisySequential(*layers).to(device)
    return net

def mnist_ONN(unitary=Unitary, num_in=N_IN, num_out=10, num_h1=256, num_h2=256, hidden_units=[256, 256], device=DEVICE, sigma_PS=0, sigma_BS=0, T0=0.03):
    """
    Creates a MLP for training on MNIST
    args:
        unitary: The type of unitary layer used (GridUnitary, FFTUnitary, etc.)
        num_h1: The number of hidden units in the first layer
        num_h2: The number of hidden units in the second layer
        device: The device to be used by torch. 'cpu' or 'cuda'
        sigma_PS: The stdev on uncertainty added to phaseshifter
        sigma_BS: The stdev on uncertainty added to beamsplitter
    returns:
        A th.nn.Sequential module with above features
    """
    f = ShiftedSoftplus(T=T0)
    layers = [
        Linear(num_in, hidden_units[0], sigma_PS=sigma_PS, sigma_BS=sigma_BS, UNet=unitary),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            Linear(nh_, nh, sigma_PS=sigma_PS, sigma_BS=sigma_BS, UNet=unitary),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        Linear(hidden_units[-1], num_out, sigma_PS=sigma_PS, sigma_BS=sigma_BS, UNet=unitary),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ])
    net = NoisySequential(*layers).to(device)
    return net

'''
Creates a QOCNN for training on MNIST.

Inputs:
    num_in: the input dimension 
    num_out: the output dimension
    hidden_units: the number of hidden units in each linear layer
    device: the device on which the network is being run
    sigma: the amount of noise
    T0: the modulation created by the nonlinearity

Outputs:
    A QOCNN satisfying the above properties
'''
def mnist_complex(num_in=N_IN, num_out=10, hidden_units=[256, 256], device=DEVICE, sigma=0, T0=0.2):
    """ f = SineModulator(T=T0)
    layers = [
        ComplexConvolution(392, filtersize = 3, stepsize = 1),
        MaxPooling(kernel_size = 2, stride = 1),
        ComplexLinear(391, hidden_units[0]),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            ComplexLinear(nh_, nh),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        ComplexLinear(hidden_units[-1], num_out),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ]) """
    #f = SineModulator(T=T0)
    f = ShiftedSoftplus(T=0.03)
    layers = [
        ComplexLinear(num_in, hidden_units[0]),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            ComplexLinear(nh_, nh),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        ComplexLinear(hidden_units[-1], num_out),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ])
    net = nn.Sequential(*layers).to(device)

    '''def to_grid_net(rand_S=True):
        grid_net = mnist_ONN(num_in=num_in, num_out=num_out, hidden_units=hidden_units)
        for lc, lo in zip(net, grid_net):
            if isinstance(lc, ComplexLinear):
                assert isinstance(lo, Linear)
                assert lc.weight.shape == (2*lo.D_out, 2*lo.D_in)
                M = lc.weight.to('cpu').data
                t0 = time()
                print(f'Converting weights of size {M.shape}')
                lo.emul_M(M, rand_S=rand_S)
                print(time()-t0)
        return grid_net'''

    # net.to_grid_net = to_grid_net
    return net

if __name__ == '__main__':

    LR_FFT = 5e-2
    LR_GRID = 2.5e-4
    LR_COMPLEX = 5e-3
    train_params = {}
    train_params['n_epochs'] = 10
    train_params['log_interval'] = 1
    train_params['batch_size'] = 32

    optim_params = {}
    optim_params['lr'] = LR_FFT
    optim_params['momentum'] = .9

    # net = mnist_grid_truncated()
    # print(net(X0))
    # train(net, **train_params, optim_params=optim_params)
    # optim_params['lr'] /= 5
    # train(net, **train_params, optim_params=optim_params)
    # th.save(net, os.path.join(DIR_TRAINED_MODELS, 'truncated_grid.pth'))
    # assert False
    # net = mnist_stacked_fft(32)
    # train(net, **train_params, optim_params=optim_params)
    # optim_params['lr'] /= 5
    # train(net, **train_params, optim_params=optim_params)
    # th.save(net, os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_1.pth'))
    # print(get_acc(net))
    net_loaded = mnist_ONN()
    net_loaded.load_state_dict(th.load(os.path.join(DIR_TRAINED_MODELS, 'grid_net.pth')))
    print("Loaded Model")
    print(get_acc(net_loaded))

    # assert False
    # train(net, **train_params, optim_params=optim_params)
    # optim_params['lr'] /= 5
    # train(net, **train_params, optim_params=optim_params)
    # th.save(net, os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_32.pth'))
    # print(get_acc(net))
    # net_loaded = th.load(os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_32.pth'))
    # print(get_acc(net_loaded))
