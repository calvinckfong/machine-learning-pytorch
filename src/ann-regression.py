import matplotlib.pylab as plot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


'''
Dataset: synthetic dataset for regression with ANN
Y ~ sin(X)
'''
class Dataset(Dataset):
    def __init__(self, train:bool):
        self.len = 1000 if train else 200
        self.X = torch.FloatTensor(size=(self.len,1)).uniform_(-5.0, 5.0)
        self.Y = torch.sin(self.X) + torch.FloatTensor(size=(self.len,1)).normal_(0, 0.1)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y
    
def Plot(train_dataset, valid_dataset, filename:str):
    plot.clf()
    plot.plot(train_dataset.X[:], train_dataset.Y[:], 'r.', label='training data')
    plot.plot(valid_dataset.X[:], valid_dataset.Y[:], 'b.', label='validation data')
    plot.savefig(filename)

'''
NN Regressor
'''
class NNRegress(nn.Module):
    def __init__(self, in_size:int, out_size:int, hidden_size:list[int]):
        super(NNRegress, self).__init__()
        self.hidden1 = nn.Linear(in_size, hidden_size[0])
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.output = nn.Linear(hidden_size[1], out_size)

    def forward(self, x):
        out = torch.sigmoid(self.hidden1(x))
        out = torch.sigmoid(self.hidden2(out))
        return self.output(out)

    #def train_step(self, batch_data):
        

if __name__ == '__main__':
    # Initialize random seeds
    torch.manual_seed(0)

    # Prepare data
    train_dataset = Dataset(train=True)
    valid_dataset = Dataset(train=False)
    Plot(train_dataset, valid_dataset, 'data.png')    
        
    model = NNRegress(1, 1, [5, 5])

    learningRate = 5e-2
    epochs = 200
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        # for x, y in train_dataset:
        #     optimizer.zero_grad()
        #     y_pred = model(x)
        #     loss = criterion(y_pred, y)
        #     train_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        optimizer.zero_grad()
        y_pred = model(train_dataset.X)
        loss = criterion(y_pred, train_dataset.Y)
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
        
        # Validation
        model.eval()
        valid_loss = 0.0
        y_pred = model(valid_dataset.X)
        valid_loss = criterion(y_pred, valid_dataset.Y).item()
        
        print(f'{epoch=} {train_loss=:.3f}, {valid_loss=:.3f}')

        # Plot last validation result
        if epoch == epochs - 1:
            plot.clf()
            plot.plot(valid_dataset.X, valid_dataset.Y, 'k.', label='GT')
            plot.plot(valid_dataset.X, y_pred.detach().numpy(), 'r.', label='prediction')
            plot.legend()
            plot.savefig('result.png')