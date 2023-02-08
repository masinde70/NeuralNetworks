import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim 


# Data transforms and loaders
#
training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),       # Data Augmentation
    transforms.ToTensor(),                        # Transforms image to range of 0 - 1
    transforms.Normalize((0.1307,), (0.3081,))    # Normalizes image
    ])

testing_transform = transforms.Compose([          # No Data Augmentation for test transform
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

trainset = datasets.<your data="" here="">('data/', download=True, train=True, transform=training_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


# Training loop
# To train your model, you need to iterate through your data batches, reset your optimizer and perform a backward step to update your weights. 
def train(model, train_loader, cost, optimizer, epoch):
  model.train()
  for e in range(epoch):
    running_loss=0
    correct=0
    for data, target in train_loader:                                 # Iterates through batches
       data = data.view(data.shape[0], -1)                           # Reshapes data
       optimizer.zero_grad()                                         # Resets gradients for new batch
       pred = model(data)                                            # Runs Forwards Pass
       loss = cost(pred, target)                                     # Calculates Loss
       running_loss+=loss 
       loss.backward()                                               # Calculates Gradients for Model Parameters
       optimizer.step()                                              # Updates Weights
       pred=pred.argmax(dim=1, keepdim=True)
       correct += pred.eq(target.view_as(pred)).sum().item()         # Checks how many correct predictions where made
  print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")


# Create a Testing Loop
def test(model, test_loader):
  model.eval()
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data = data.view(data.shape[0], -1)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} = {100*(correct/len(test_loader.dataset))}%)')  