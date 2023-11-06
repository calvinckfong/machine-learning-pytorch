import math
import matplotlib.pylab as plot
from numpy import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

'''
Dataset: A synthetic dataset for multivariate regression
Y = a1*X1 + a2*X2 + a3*X3 + b
'''

class Dataset(Dataset):
    # define parameters of the linear regression to be determined
    a1 = 3.0
    a2 = -0.2
    a3 = 1.5
    b = -0.75
    
    def __init__(self, train:bool):
        if train:
            self.len = 1000
        else:
            self.len = 200
            
        self.X = torch.from_numpy(random.uniform(-1, 1, (self.len, 3)))
        self.Y = self.a1 * self.X[:,0] \
            + self.a2 * self.X[:,1] \
            + self.a3 * self.X[:,2] \
            + self.b + random.normal(loc=0.0, scale=0.5, size=(self.len))  
            
        # make input and output as float
        self.X = self.X.float().view(-1, 3)
        self.Y = self.Y.float().view(-1, 1)

    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index):
        x = self.X[index,:]
        y = self.Y[index]
        return x, y
    
'''
Multivariate Regressor
'''    
class MultiRegress(nn.Module):
    def __init__(self, in_size, out_size):
        super(MultiRegress, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        return self.linear(x)
    
    
if __name__ == '__main__':
    # Initialize random seeds
    torch.manual_seed(0)
    random.seed(0)
    
    # Prepare datasets
    train_dataset = Dataset(train=True)
    valid_dataset = Dataset(train=False)
    print(train_dataset.a1, train_dataset.a2, train_dataset.a3, train_dataset.b)
        
    learningRate = 5e-3
    epochs = 50
    model = MultiRegress(3, 1)
    print(model.state_dict())
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    train_loss_list = []
    valid_loss_list = []
    best_param = None
    min_valid_loss = 1.0e10
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for index, (x, y) in enumerate(train_dataset):
            x_ = Variable(x, requires_grad=True)
            y_ = Variable(y.unsqueeze(1))
            optimizer.zero_grad()
            y_pred = model(x_)
            loss = criterion(y_, y_pred)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        model.eval()
        valid_loss = 0.0
        for (x, y) in valid_dataset:
            y_pred = model(x)
            valid_loss += criterion(y, y_pred).item()
            
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_param = model.state_dict()
        else:
            # reduce learningRate if no improvement
            learningRate *= 0.6
            for g in optimizer.param_groups:
                g['lr'] = learningRate        
                        
        train_loss /= train_dataset.len
        valid_loss /= valid_dataset.len
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        print(f'{epoch=:3d} {train_loss=:.6f} {valid_loss=:.6f} {learningRate=:.6f}')

        # early termination
        if learningRate < 1e-6:
            break

    print(best_param)
    
    plot.semilogy(train_loss_list, 'b')
    plot.semilogy(valid_loss_list, 'r')
    plot.savefig('loss.png')