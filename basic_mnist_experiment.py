import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets

## ==================== MODELS ====================

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


## ==================== UTILITY ====================

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.g = generator
        self.d = discriminator

    def forward(self, z):
        x_fake = self.g(z)
        y = self.d(x_fake)
        return y

    def sample(self, latent_dim, num_samples):
        z = self.sample_latent(latent_dim, num_samples)
        with torch.no_grad():
            fake = self.g(z)
        fake = fake.numpy()
        return fake

    def sample_latent(self, latent_dim, num_samples):
        z = np.random.normal(size=(num_samples, latent_dim))
        z = torch.from_numpy(z).float()
        return z
        
def sample_gan_25(gan, latent_dim, scale=5):
    canvas = np.zeros((5 * 28, 5 * 28), dtype=np.uint8)
    samples = gan.sample(latent_dim, 25)
    for i in range(5):
        for j in range(5):
            canvas[28*i:28*i+28,28*j:28*j+28] = \
                (255 * samples[5*i + j].reshape((28,28))).astype(np.uint8)
    return canvas.repeat(scale, axis=0).repeat(scale, axis=1)

## ==================== TRAINING ====================

import random # for sampling
from PIL import Image # for writing benchmarks
random.seed(12345)

mnist = datasets.MNIST("../data", train=True, download=True)
mnist_x = mnist.data

def sample_mnist(minibatch_size):
    x = mnist_x[random.sample(range(mnist_x.shape[0]), minibatch_size)]
    x = (x.reshape((-1, 784)) / 255.).float()
    return x

EPSILON = 1e-7

g = MNISTFullyConnectedGenerator(latent_dim=32)
d = MNISTFullyConnectedDiscriminator()

g.train()
d.train()

gan = GAN(g, d)

MB_SIZE = 64 # Minibatch size

g_opt = optim.SGD(g.parameters(), lr=0.005)
d_opt = optim.SGD(d.parameters(), lr=0.01)

NUM_ITERS = 100
NUM_MBS = 10

k = 2
for num_it in range(NUM_ITERS):
    for mb_num in range(NUM_MBS):
        for i in range(k):
            d_opt.zero_grad()
            
            z = gan.sample_latent(32, MB_SIZE)
            x = sample_mnist(MB_SIZE)

            objective = torch.log(EPSILON + d(x)) + torch.log(EPSILON + 1 - gan(z))
            # Discriminator's goal is to maximize, so we negate objective
            # function, since pytorch optimizers minimize.
            objective = -torch.sum(objective)
            #print("D objective:",objective)
            objective.backward()

            d_opt.step()


        g_opt.zero_grad()
        
        z = gan.sample_latent(32, MB_SIZE)
        
        ## VANILLA LOSS:
        #objective = torch.sum(torch.log(EPSILON + 1 - gan(z)))
        ## UNSATURAING LOSS:
        objective = -torch.sum(torch.log(EPSILON + gan(z)))
        
        objective.backward()
        
        g_opt.step()
        

    ## estimate the minimax expression
    with torch.no_grad():
        x = sample_mnist(MB_SIZE)
        z = gan.sample_latent(32, MB_SIZE)
        value = torch.mean(torch.log(EPSILON + d(x))) + torch.mean(torch.log(EPSILON + 1 - gan(z)))
        print("value is",value)

    im = Image.fromarray(sample_gan_25(gan, 32))
    im.save("run_samples/iteration_%03d.png" % num_it)
