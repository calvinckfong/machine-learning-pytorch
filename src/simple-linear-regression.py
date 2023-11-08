import math
import matplotlib.pylab as plot
from numpy import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

'''
Dataset: A synthetic data set for linear regression
Y = w*X + b
'''
class Dataset(Dataset):
    # define parameters of the linear regression to be determined
    w = math.pi
    b = -math.exp(1.0)
    
    def __init__(self, train):
        if train:
            # training set
            self.len = 4000
            self.X = torch.linspace(-5, 5, self.len)
            self.Y = self.w * self.X + self.b + random.normal(loc=0.0, scale=1.0, size=(self.len))
        else:
            # test/validation set
            self.len = 1000
            self.X = torch.from_numpy(random.uniform(-5, 5, (self.len)))
            self.Y = self.w * self.X + self.b + random.normal(loc=0.0, scale=1.0, size=(self.len))

        # make input and output as float
        self.X = self.X.float()
        self.Y = self.Y.float()
        
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

'''
Linear Regressor
'''
class LinearRegress(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegress, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        return self.linear(x)
        
    def train_step(self, epoch, train_dataset, optimizer, criterion):
        model.train()
        train_loss = 0.0
        for x, y in train_dataset:
            optimizer.zero_grad()
            y_pred = model(x.view(-1,1))
            loss = criterion(y, y_pred)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()        
        
        return train_loss / train_dataset.len

    def validation_step(self, epoch, valid_dataset, plot_prediction):
        model.eval()
        x = valid_dataset.X
        y = valid_dataset.Y    
        y_pred = model(x.view(-1,1)).squeeze()
        valid_loss = criterion(y, y_pred).item()
            
        if plot_prediction:
            plot.clf()
            plot.plot(x, y, 'b.', label='GT')
            plot.plot(x, y_pred.detach().numpy(), 'r.', label='Prediction')
            plot.title(f'Epoch = {epoch}')
            plot.legend()
            plot.savefig('pred.png')
        
        return valid_loss

def plot_loss(train_loss, valid_loss, filename):
    plot.clf()
    plot.plot(train_loss, 'b', label='Training')
    plot.plot(valid_loss, 'r', label='Validation')
    plot.legend()
    plot.savefig(filename)

if __name__ == '__main__':
    # Initialize random seeds
    torch.manual_seed(0)
    random.seed(0)
    
    train_dataset = Dataset(train=True)
    valid_dataset = Dataset(train=False)
    
    learningRate = 1e-3
    epochs = 20
    model = LinearRegress(1, 1)
    print(model.state_dict())
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    
    train_loss_list = []
    valid_loss_list = []
    best_param = None
    min_valid_loss = 1.0e10
    for epoch in range(epochs):
        # Training
        train_loss = model.train_step(epoch, train_dataset, optimizer, criterion)
        
        # Validation
        valid_loss = model.validation_step(epoch, valid_dataset, (epoch==epochs-1))

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        print(f'{epoch=:3d} {train_loss=:.6f} {valid_loss=:.6f} {learningRate=:.6f}')

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_param = model.state_dict()
        else:
            # reduce learningRate if no improvement
            learningRate *= 0.6
            for g in optimizer.param_groups:
                g['lr'] = learningRate
                        
        # early termination
        if learningRate < 1e-6:
            print('Early Termination')
            # Valid and plot the prediction
            model.validation_step(epoch, valid_dataset, True)
            break
    
    print(f'{min_valid_loss=}')
    print(best_param)
    
    plot_loss(train_loss_list, valid_loss_list, 'loss.png')
    
