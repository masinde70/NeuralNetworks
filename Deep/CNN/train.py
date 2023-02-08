import torch 
import torch.nn as nn
import torch.nn.functional as F 

def train(model, train_loader, cost, optimizer, epoch):
    model.train()
    for e in range(epoch):
        running_loss = 0
        correct=0
        for data, target in train_loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = cost(pred, target)
            running_loss += loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax()
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")    