SEED = 12345

import numpy as np
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector

from models import *
from util import *
from datasets import *

import json
import os

DATASET = MNIST(shape="flat")
RESULTS_DIR = "runs/run1"

VERBOSE = False

EPSILON = 1e-7
LATENT_DIM = 32
NUM_ITERS = 100
NUM_MBS = 20
MB_SIZE = 64 # Minibatch size

g = MNISTFullyConnectedGenerator(latent_dim=LATENT_DIM)
d = MNISTFullyConnectedDiscriminator()

g.train()
d.train()

G_LR = 0.05
D_LR = 0.2

g_opt = optim.SGD(g.parameters(), lr=G_LR)
d_opt = optim.SGD(d.parameters(), lr=D_LR)

k = 1

training_logs = {}
training_logs["iteration"] = []
training_logs["v"] = []
training_logs["d_real_acc"] = []
training_logs["d_fake_acc"] = []
training_logs["d_grad_norm"] = []
training_logs["g_grad_norm"] = []

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# Used for consistent samples
sampling_vec = sample_latent_prior(LATENT_DIM, 64)

# Do NUM_ITERS iterations
for num_it in range(NUM_ITERS):
    if VERBOSE:
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
            x = DATASET.sample_train(MB_SIZE)

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
        x = DATASET.sample_train(MB_SIZE)
        z = sample_latent_prior(LATENT_DIM, MB_SIZE)
        value = torch.mean(torch.log(EPSILON + d(x))) + torch.mean(torch.log(EPSILON + 1 - d(g(z))))
        value = value.item()
        d_real_acc = (d(x) > 0.5).float().mean().item()
        d_fake_acc = (d(g(z)) < 0.5).float().mean().item()

    if VERBOSE:
        print(f"V(D,G) = {value}")
        print(f"D(x) acc = {d_real_acc}")
        print(f"D(g(z)) acc = {d_fake_acc}")

    training_logs["iteration"].append(num_it)
    training_logs["v"].append(value)
    training_logs["d_real_acc"].append(d_real_acc)
    training_logs["d_fake_acc"].append(d_fake_acc)
    training_logs["d_grad_norm"].append(d_grad_norm)
    training_logs["g_grad_norm"].append(g_grad_norm)

    with torch.no_grad():
        sampled_ims = g(sampling_vec).numpy().reshape(-1, 28, 28)
    im = convert_to_image(tile_images(sampled_ims, (8,8), 4))
    im.save(os.path.join(RESULTS_DIR, "iteration_%03d.png" % num_it))

with open(os.path.join(RESULTS_DIR, "logs.csv"), "w") as f:
    keys = list(training_logs.keys())
    f.write(",".join(keys) + "\n")
    for i in range(len(training_logs["iteration"])):
        f.write(",".join(str(training_logs[key][i]) for key in keys) + "\n")

with open(os.path.join(RESULTS_DIR, "hyperparams.json"), "w") as f:
    json.dump({
        "DATASET" : str(DATASET),
        "LATENT_DIM" : LATENT_DIM,
        "NUM_ITERS" : NUM_ITERS,
        "NUM_MBS" : NUM_MBS,
        "MB_SIZE" : MB_SIZE,
        "G_LR" : G_LR,
        "D_LR" : D_LR
        }, f)