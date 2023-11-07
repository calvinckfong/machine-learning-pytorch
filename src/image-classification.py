import matplotlib.pylab as plot
from matplotlib.patches import Rectangle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.models as models

class ImageClassifier(nn.Module):
    in_size = 28*28 # images of 28x28
    out_size = 10   # 10 digits
    
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.resnet = models.resnet18(self.in_size)
        # disable resnet gradient update for resnet except conv1
        for param in self.resnet.parameters():
            param.requires_grad = False
        # change first conv1 to 1-channel for grayscale image
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = nn.Linear(1000, self.out_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.resnet(x)
        out = self.fc(out)
        return self.softmax(out)

def Plot(image, filename):
    plot.clf()
    plot.imshow(image.squeeze().numpy(), cmap='gray', vmin=0, vmax=255)
    plot.savefig(filename)

def DrawResult(epoch, model, valid_loader, filename):
    images, targets = next(iter(valid_loader))
    out = model(images.to(device))
    pred = out.data.max(1)[1]
    plot.clf()
    plot.title(f'Epoch {epoch}')
    plot.axis('off')
    for i in range(8):
        subtitle = f'{pred[i].item()}[{targets[i].item()}]'
        plot.subplot(2, 4, i+1, title=subtitle)
        plot.imshow(images[i].squeeze().numpy())
        plot.axis('off')
    plot.savefig(filename)
        
    

def PrintModel(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}, {param.data}')
    

if __name__ == '__main__':
    # select device
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    
    # Prepare Data
    root = './dataset/MNIST'
    transform = transforms.Compose([
                    transforms.
                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                    transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                ])
    train_dataset = MNIST(root, train=True, download=True, transform=transform)
    valid_dataset = MNIST(root, train=False, download=False, transform=transform)
    train_datasize = train_dataset.__len__()
    valid_datasize = valid_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1000)
    
    # for index, (image, target) in enumerate(train_loader):
    #     print(f'{index=}\n{target.item()=}')
    #     Plot(image, 'test.png')
    #     break
    
    model = ImageClassifier().to(device)
    #PrintModel(model)
    
    learningRate = 1e-1
    epochs = 50
    #criterion = nn.MSELoss()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    best_accuracy = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for index, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(images.to(device))
            loss = criterion(y_pred, targets.to(device))
            loss.backward()
            train_loss += torch.sum(loss).item()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        for (images, targets) in valid_loader:
            out = model(images.to(device))
            pred = out.data.max(1)[1]
            correct += (pred == targets.to(device)).cpu().sum()
        accuracy = (correct / valid_datasize).item()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
             # reduce learningRate if no improvement
            learningRate *= 0.6
            for g in optimizer.param_groups:
                g['lr'] = learningRate                   
        
        print(f'{epoch=:3d} {train_loss=:.3f} {accuracy=:.6f} {learningRate=:.6f}')
        #PrintModel(model)
        
        DrawResult(epoch, model, valid_loader, 'result.png')
        
        # early termination
        if learningRate < 1e-4:
            print('Early termination')
            break

    