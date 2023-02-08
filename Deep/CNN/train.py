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


    class Model(nn.Module):
        def __init__(self):
            super.__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            