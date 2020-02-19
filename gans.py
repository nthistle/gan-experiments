import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import parameters_to_vector
from models import *
from torchvision import datasets

## ==================== UTILITY ====================

# Samples from N(0,1)^{latent_dim}
def sample_latent_prior(latent_dim, batch_size):
    z = np.random.normal(size=(batch_size, latent_dim))
    return torch.from_numpy(z).float()

# Does not preserve gradient; ONLY use this for visualization purposes
def sample_generator(generator, latent_dim, batch_size):
    z = sample_latent_prior(latent_dim, batch_size)
    with torch.no_grad():
        fake = generator(z)
    return fake.numpy()

# Hardcoded method to sample 25 images from p_g(x), and arrange them
# in a 5x5 tile.
def sample_gan_25(generator, latent_dim, scale=5):
    canvas = np.zeros((5 * 28, 5 * 28), dtype=np.uint8)
    samples = sample_generator(g, latent_dim, 25)
    for i in range(5):
        for j in range(5):
            canvas[28*i:28*i+28,28*j:28*j+28] = \
                (255 * samples[5*i + j].reshape((28,28))).astype(np.uint8)
    return canvas.repeat(scale, axis=0).repeat(scale, axis=1)

## TODO: confirm that this is properly normalized for # of parameters...
def get_gradient_norm(parameters):
    grads = []
    for param in parameters:
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return torch.norm(grads, p=2)

## ==================== TRAINING ====================

import random # for sampling
from PIL import Image # for writing benchmarks
import os
random.seed(12345)

if not os.path.exists("run_samples"):
    os.mkdir("run_samples")

mnist = datasets.MNIST("../data", train=True, download=True)
mnist_x = mnist.data

def sample_mnist(minibatch_size):
    x = mnist_x[random.sample(range(mnist_x.shape[0]), minibatch_size)]
    x = (x.reshape((-1, 784)) / 255.).float()
    return x

EPSILON = 1e-7
LATENT_DIM = 32
NUM_ITERS = 100
NUM_MBS = 20

g = MNISTFullyConnectedGenerator(latent_dim=LATENT_DIM)
d = MNISTFullyConnectedDiscriminator()

g.train()
d.train()

MB_SIZE = 64 # Minibatch size

g_opt = optim.SGD(g.parameters(), lr=0.05)
d_opt = optim.SGD(d.parameters(), lr=0.2)

k = 1

training_logs = {}
training_logs["iteration"] = []
training_logs["v"] = []
training_logs["d_real_acc"] = []
training_logs["d_fake_acc"] = []
training_logs["d_grad_norm"] = []
training_logs["g_grad_norm"] = []

# Do NUM_ITERS iterations
for num_it in range(NUM_ITERS):
    print()
    print(f"Iteration {num_it+1}/{NUM_ITERS}")

    ## TODO: Find better way to record/average these (?)
    d_grad_norm = 0.0 
    g_grad_norm = 0.0

    # With NUM_MBS minibatches each
    for mb_num in range(NUM_MBS):

        # Do k steps of optimizing d
        for i in range(k):
            # Reset discriminator gradients
            d_opt.zero_grad()

            # Sampling from p_g(x) by sampling z and then evaluating g(z)
            z = sample_latent_prior(LATENT_DIM, MB_SIZE)

            # We don't need gradient through generator when training D
            with torch.no_grad():
                x_fake = g(z)

            # Sampling from p_data(x) by taking a random sample from our dataset
            x = sample_mnist(MB_SIZE)

            # Objective function is V(D,G)
            objective = torch.log(EPSILON + d(x)) + torch.log(EPSILON + 1 - d(x_fake))

            # Discriminator's goal is to maximize, so we negate objective
            # function, since pytorch optimizers minimize.
            objective = -torch.mean(objective)

            # Backpropagate and step discriminator optimizer
            objective.backward()
            d_opt.step()

            # Record gradient magnitudes
            with torch.no_grad():
                d_grad_norm += get_gradient_norm(d.parameters()).item() / (k * NUM_MBS)


        # Reset generator gradients
        g_opt.zero_grad()
        
        # Sampling from latent prior
        z = sample_latent_prior(LATENT_DIM, MB_SIZE)
        
        # Unsaturaing loss for optimizing G -- recommended in original gan paper,
        # as gradients of vanila loss ( log( 1 - d(g(z)) ) ) are initially sparse.
        objective = -torch.mean(torch.log(EPSILON + d(g(z))))

        # Backpropagate and step generator optimizer
        objective.backward()
        g_opt.step()

        # Record gradient magnitudes
        with torch.no_grad():
            g_grad_norm += get_gradient_norm(g.parameters()).item() / NUM_MBS
        

    # Estimate the value of V(D,G)
    with torch.no_grad():
        x = sample_mnist(MB_SIZE)
        z = sample_latent_prior(LATENT_DIM, MB_SIZE)
        value = torch.mean(torch.log(EPSILON + d(x))) + torch.mean(torch.log(EPSILON + 1 - d(g(z))))
        value = value.item()
        d_real_acc = (d(x) > 0.5).float().mean().item()
        d_fake_acc = (d(g(z)) < 0.5).float().mean().item()

    print(f"V(D,G) = {value}")
    print(f"D(x) acc = {d_real_acc}")
    print(f"D(g(z)) acc = {d_fake_acc}")

    training_logs["iteration"].append(num_it)
    training_logs["v"].append(value)
    training_logs["d_real_acc"].append(d_real_acc)
    training_logs["d_fake_acc"].append(d_fake_acc)
    training_logs["d_grad_norm"].append(d_grad_norm)
    training_logs["g_grad_norm"].append(g_grad_norm)

    im = Image.fromarray(sample_gan_25(g, LATENT_DIM))
    im.save("run_samples/iteration_%03d.png" % num_it)

with open("logs.csv", "w") as f:
    keys = list(training_logs.keys())
    f.write(",".join(keys) + "\n")
    for i in range(len(training_logs["iteration"])):
        f.write(",".join(str(training_logs[key][i]) for key in keys) + "\n")
