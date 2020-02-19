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