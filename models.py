import torch
import torch.nn as nn
import torch.nn.functional as F

# 79216 parameters, 2 fully connected layers
class MNISTFullyConnectedGenerator(nn.Module):
    def __init__(self, latent_dim=32):
        super(MNISTFullyConnectedGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 96)
        self.fc2 = nn.Linear(96, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 157029 parameters, 3 fully connected layers, with significant dropout
class MNISTFullyConnectedDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTFullyConnectedDiscriminator, self).__init__()
        self.fc1 = nn.Linear(784, 196)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(196, 16)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        x = self.drop2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class GMMSimpleGenerator(nn.Module):
    def __init__(self, latent_dim=4, n_dim=2):
        super(GMMSimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, (latent_dim+n_dim)//2)
        self.fc2 = nn.Linear((latent_dim+n_dim)//2, n_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class GMMSimpleDiscriminator(nn.Module):
    def __init__(self, n_dim=2):
        super(GMMSimpleDiscriminator, self).__init__()
        self.fc1 = nn.Linear(n_dim, 3 * n_dim)
        self.fc2 = nn.Linear(3 * n_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class GMMDenseGenerator(nn.Module):
    def __init__(self, latent_dim=256):
        super(GMMDenseGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc3(x)
        return x

class GMMDenseDiscriminator(nn.Module):
    def __init__(self):
        super(GMMDenseDiscriminator, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x