from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import yaml
from dotmap import DotMap
import os
import neptune
import libs.checkpoints as checkpoints
from PIL import Image


def getSupportedDevice():
    if torch.cuda.is_available():        
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = getSupportedDevice()
BASE_TRANSFORMATIONS = [ transforms.ToTensor(),  transforms.Normalize((0.1307,), (0.3081,))]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def __startTraining(args, model, device, train_loader, optimizer, epoch, run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, predicted = torch.max(output, 1)
        correctPredictions = (predicted == target).sum().item()
        totalSamples = len(predicted)
        run['train/accuracy'].append(100*correctPredictions/totalSamples) 
        loss = F.nll_loss(output, target)
        loss.backward()
        run['train/loss'].append(loss.item())
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def train(config ):
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--config', default='initial_experiment.yaml', metavar='C',
    #                     help='input config file for experiment (default: initial_experiment.yaml)')
   
    checkpoint = checkpoints.loadLastCheckpoint()
    run = neptune.init_run()

    run['config'] = config
    
    torch.manual_seed(42)

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.batch_size}

    if torch.cuda.is_available():
        
        device = torch.device("cuda")
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    run['device'] = device.type

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./dataset', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./dataset', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    loss = 0
    previousEpoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint['modelState'])
        optimizer.load_state_dict(checkpoint['optimizerState'])
        previousEpoch = checkpoint['epoch']+1

    for epoch in range(previousEpoch, config.max_epochs + previousEpoch):
        __startTraining(config, model, device, train_loader, optimizer, epoch, run)
        loss = test(model, device, test_loader)
        run['epoch'] = epoch
        run['loss'] = loss
        newCheckpoint = {
            'modelState' : model.state_dict(),
            'optimizerState' : optimizer.state_dict(),
            'epoch' : epoch
        }
        checkpoints.saveCheckpoint(run, newCheckpoint)
        scheduler.step()

    run.stop()





def infer(image, modeState):

    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28))  
    ]+ BASE_TRANSFORMATIONS)
    
    input_tensor = transform(image).unsqueeze(0)

    model = Net().to(DEVICE)
    model.load_state_dict(modeState)
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)

    return torch.argmax(output, dim=1).item()
 
