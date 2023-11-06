import math
import matplotlib.pylab as plot
from numpy import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

'''
Dataset: A synthetic dataset for polynomial regression
Y = a1*X + a2*X^2 + a3*X^2 + b
'''
class Dataset(Dataset):
    # define parameters of the linear regression to be determined
    a1 = math.pi
    a2 = -0.2 * math.pi
    a3 = -0.05 * math.pi
    b = -math.exp(1.0)

    def __init__(self, train:bool):
        if train:
            # training set
            self.len = 1000
            self.X = torch.linspace(-1, 1, self.len)
        else:
            # training set
            self.len = 200
            self.X = torch.from_numpy(random.uniform(-1, 1, (self.len)))
            
        self.Y = self.a1 * self.X \
            + self.a2 * torch.pow(self.X, 2) \
            + self.a3 * torch.pow(self.X, 3) \
            + self.b + random.normal(loc=0.0, scale=0.5, size=(self.len))

        # make input and output as float
        self.X = self.X.float().view(-1, 1)
        self.Y = self.Y.float().view(-1, 1)
        
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

'''
Polynomial Regressor
'''
class PolyRegress(nn.Module):
    def __init__(self, in_size, out_size):
        super(PolyRegress, self).__init__()
        self.linear = nn.Linear(3*in_size, out_size) # third-order polynomial
        
    def forward(self, x):
        x_ = self.poly_feature(x)
        #print(x_.shape)
        return self.linear(x_)
    
    def poly_feature(self, x):
        x_ = x.unsqueeze(1)
        return torch.cat([x_, x_**2, x_**3],1).view(1, -1)
    
if __name__ == '__main__':
    # Initialize random seeds
    torch.manual_seed(0)
    random.seed(0)
    
    # Prepare datasets
    train_dataset = Dataset(train=True)
    valid_dataset = Dataset(train=False)
    print(train_dataset.a1, train_dataset.a2, train_dataset.a3, train_dataset.b)
    
    plot.plot(train_dataset.X, train_dataset.Y, 'b')
    plot.plot(valid_dataset.X, valid_dataset.Y, 'r.')
    plot.savefig('dataset.png')
    
    learningRate = 5e-3
    epochs = 100
    model = PolyRegress(1, 1)
    print(model.state_dict())
    criterion = torch.nn.MSELoss() 
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
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

        # Validation
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
