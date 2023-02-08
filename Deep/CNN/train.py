import torch 
import torch.nn as nn
import torch.nn.functional as F 

def train(model, train_loader, cost, optimizer, epoch):
    model.train()
    for e in range(epoch):
        running_loss = 0
        