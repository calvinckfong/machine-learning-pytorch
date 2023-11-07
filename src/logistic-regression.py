import matplotlib.pylab as plot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

'''
Dataset: synthetic dataset for classification
'''
class Dataset(Dataset):
    # define the centers and the standard deviations of the clusters
    centers = torch.Tensor([
        [-1.0, 1.0],
        [1.0, -1.0]
    ])
    stdevs = torch.Tensor([
        2.0,
        2.0
    ])
    
    def __init__(self, train:bool):
        if train:
            self.len = 1000
        else:
            self.len = 200
        
        halfway = self.len//2
        self.Y = torch.tensor([0] * self.len)
        input = (np.random.random((self.len, 2)) - 0.5)*3
        input[:halfway] += self.centers[0].numpy()
        input[halfway:] += self.centers[1].numpy()
        self.Y[:halfway] = torch.tensor(0)
        self.Y[halfway:] = torch.tensor(1)
        self.X = torch.from_numpy(input).float()
        self.Y = self.Y.float()
        
        # self.Y = torch.randint(0, 2, (self.len, 1))
        # X = []
        # for y in self.Y:
        #     x = (torch.rand((1, 2))-0.5)*self.stdevs[y.item()]  + self.centers[y.item()]
        #     X.append(x)
        # self.X = torch.cat(X, dim=0)
        #print(self.X)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = self.X[index,:].view(1,-1)
        #print(x.shape)
        y = self.Y[index].view(-1,1)
        return x, y
    
'''
Logistic Regressor
'''
class LogisticRegress(nn.Module):
    def __init__(self, in_size, out_size):
        super(LogisticRegress, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        y = self.linear(x)
        #print(f'y={y[:10,:]}')
        return torch.sigmoid(y)

def Plot(dataset:Dataset, filename:str):
    mask0 = torch.nonzero(dataset.Y.view(1,-1) == 0)[:,1]
    mask1 = torch.nonzero(dataset.Y.view(1,-1) == 1)[:,1]
    class0 = dataset.X[mask0,:]
    class1 = dataset.X[mask1,:]
    # print(class0)
    # print(class1)
    plot.clf()
    plot.plot(class0[:,0], class0[:,1], 'r.', label='Class 0')
    plot.plot(class1[:,0], class1[:,1], 'b.', label='Class 1')
    plot.savefig(filename)

if __name__ == '__main__':
    # Initialize random seeds
    torch.manual_seed(0)

    # Prepare data
    train_dataset = Dataset(train=True)
    valid_dataset = Dataset(train=False)
    Plot(train_dataset, 'train_dataset.png')
    Plot(valid_dataset, 'valid_dataset.png')
    # print(valid_dataset.X.shape)
    # c0_idx = (valid_dataset.Y == 0).nonzero()[:,0]
    # c1_idx = (valid_dataset.Y == 1).nonzero()[:,0]
    # c0 = valid_dataset.X[c0_idx]
    # c1 = train_dataset.X[c1_idx]
    
    # plot.plot(c0[:, 0], c0[:, 1], 'r.', label='Class 0')
    # plot.plot(c1[:, 0], c1[:, 1], 'b.', label='Class 1')
    # plot.legend()
    # plot.savefig('data.png')
    
    model = LogisticRegress(2, 1)
    
    learningRate = 1e-3
    epochs = 20
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_dataset:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        valid_loss = 0.0
        y_pred = torch.round(model(valid_dataset.X))
        # print(y_pred[:10])
        # print(valid_dataset.Y[:10].view(-1,1))
        # print(torch.sum(y_pred == valid_dataset.Y.view(-1,1)))
        correct = torch.sum(y_pred == valid_dataset.Y.view(-1,1))
        accuracy = correct / valid_dataset.len
        
        print(f'{epoch=} {train_loss=:.3f} validation accuracy {accuracy:.3f}')
        # break
        
    print(model.state_dict())
    y_pred = torch.round(model(valid_dataset.X))
    c0_idx = (y_pred == 0).nonzero()[:,0]
    c1_idx = (y_pred == 1).nonzero()[:,0]
    c0 = valid_dataset.X[c0_idx,:]
    c1 = valid_dataset.X[c1_idx,:]
    plot.clf()
    plot.plot(c0[:,0], c0[:,1], 'r.', label='Class 0')
    plot.plot(c1[:,0], c1[:,1], 'b.', label='Class 1')
    plot.plot()
    plot.savefig('result.png')           
    