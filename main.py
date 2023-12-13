#%%
import random

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import copy

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 36

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
#model = NeuralNetwork()
print(model)

lr = 1e-3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

weight_mat = torch.rand(168, 168)
print(weight_mat)


def sample_new_mat(mat):
    ret = copy.deepcopy(mat)
    for t in range(60):
        i = np.random.randint(0, ret.shape[0])
        j = np.random.randint(0, ret.shape[1])
        if np.random.rand()<0.5:
            ret[i][j] += lr
        else:
            ret[i][j] -= lr
    return ret


def get_new_mat(mat, X, y):
    ret = mat
    new_mat = mat
    best_loss = 999999
    with torch.no_grad():
        for i in range(6):
            new_x = copy.deepcopy(X)
            new_x = torch.reshape(new_x, (168, 168))
            new_x = torch.matmul(new_x, new_mat)
            new_x = torch.reshape(new_x, (36, 1, 28, 28))
            pred = model(X)
            new_loss = loss_fn(pred, y).item()
            if(new_loss < best_loss):
                best_loss = new_loss
                ret = new_mat
            new_mat = sample_new_mat(mat)
    return ret


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    w = weight_mat
    for batch, (X, y) in enumerate(dataloader):
        # X = torch.reshape(X, (168, 168))
        # X = torch.matmul(X, w)
        # X = torch.reshape(X, (36, 1, 28, 28))
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #w = get_new_mat(w, X, y)


        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        w = weight_mat
        for X, y in dataloader:
            # X = torch.reshape(X, (168, 168))
            # X = torch.matmul(X, w)
            # X = torch.reshape(X, (36, 1, 28, 28))
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#%%

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")