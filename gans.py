SEED = 12345

import numpy as np
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from sklearn.decomposition import PCA

from models import *
from util import *
from datasets import *

import json
import os
import pickle

DATASET = Grid2D()
RESULTS_DIR = "runs/grid2d_run1"

G_POINT_SAVE_FILENAME = "generator_saved_points.pkl"

D_POINT_REPR_SIZE = 500 #256 #16384
D_POINT_REPR_PTS = DATASET.sample_train(D_POINT_REPR_SIZE)

G_POINT_REPR_PER = 5 # how many points to save from G per iteration

VERBOSE = False
GENERATE_PLOTS = True
SHOW_PLOTS = True

EPSILON = 1e-7
LATENT_DIM = 32 #32
NUM_ITERS = 100
NUM_MBS = 20
MB_SIZE = 128 # Minibatch size

g = GMMDenseGenerator(latent_dim=LATENT_DIM) #MNISTFullyConnectedGenerator(latent_dim=LATENT_DIM)
d = GMMDenseDiscriminator() #MNISTFullyConnectedDiscriminator()

G_LR = 0.01
D_LR = 0.01

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

discriminator_reprs = []

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

if os.path.exists(os.path.join(RESULTS_DIR, G_POINT_SAVE_FILENAME)):
    with open(os.path.join(RESULTS_DIR, G_POINT_SAVE_FILENAME), "rb") as f:
        G_POINT_REPR_PTS = pickle.load(f)
    REPR_PTS = torch.from_numpy(np.concatenate([D_POINT_REPR_PTS, G_POINT_REPR_PTS]))
    SAVE_G_PTS = False
else:
    print("Warning: did not find saved generator samples for point representation!")
    G_POINT_REPR_PTS = []
    SAVE_G_PTS = True
    REPR_PTS = D_POINT_REPR_PTS

# Used for consistent samples
sampling_vec = sample_latent_prior(LATENT_DIM, 64)

# Do NUM_ITERS iterations
for num_it in range(NUM_ITERS):
    if VERBOSE:
        print()
    print(f"Iteration {num_it+1}/{NUM_ITERS}")

    g.train()
    d.train()

    ## TODO: Find better way to record/average these (?)
    d_grad_norms = []
    g_grad_norms = []

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
                d_grad_norms.append(get_gradient_norm(d.parameters()).item() / (k * NUM_MBS))


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
            g_grad_norms.append(get_gradient_norm(g.parameters()).item() / NUM_MBS)


    g.eval()
    d.eval()

    # Estimate the value of V(D,G)
    with torch.no_grad():
        x = DATASET.sample_train(MB_SIZE)
        z = sample_latent_prior(LATENT_DIM, MB_SIZE)
        value = ((torch.log(EPSILON + d(x))) + torch.mean(torch.log(EPSILON + 1 - d(g(z))))).numpy()[:,0]
        d_real_acc = (d(x) > 0.5).float().numpy()[:,0]
        d_fake_acc = (d(g(z)) < 0.5).float().numpy()[:,0]

        discriminator_reprs.append(d(REPR_PTS).numpy())

    if VERBOSE:
        print(f"V(D,G) = {value}")
        print(f"D(x) acc = {d_real_acc}")
        print(f"D(g(z)) acc = {d_fake_acc}")

    training_logs["iteration"].append(num_it)
    training_logs["v"].append(value)
    training_logs["d_real_acc"].append(d_real_acc)
    training_logs["d_fake_acc"].append(d_fake_acc)
    training_logs["d_grad_norm"].append(d_grad_norms)
    training_logs["g_grad_norm"].append(g_grad_norms)

    with torch.no_grad():
        save_sample = g(sample_latent_prior(LATENT_DIM, G_POINT_REPR_PER)).numpy()
        if SAVE_G_PTS:
            G_POINT_REPR_PTS.append(save_sample)
        sampled_ims = g(sampling_vec).numpy()

    DATASET.write_sample(sampled_ims, os.path.join(RESULTS_DIR, "iteration_%03d" % num_it), consistent=True)

with open(os.path.join(RESULTS_DIR, "logs.csv"), "w") as f:
    keys = list(training_logs.keys())
    f.write(",".join(keys) + "\n")
    for i in range(len(training_logs["iteration"])):
        def get_item(key):
            item = training_logs[key][i]
            if hasattr(item, "__len__"):
                item = sum(item) / len(item) # write the average
            return item
        f.write(",".join(str(get_item(key)) for key in keys) + "\n")

with open(os.path.join(RESULTS_DIR, "hyperparams.json"), "w") as f:
    json.dump({
        "DATASET" : str(DATASET),
        "LATENT_DIM" : LATENT_DIM,
        "NUM_ITERS" : NUM_ITERS,
        "NUM_MBS" : NUM_MBS,
        "MB_SIZE" : MB_SIZE,
        "G_LR" : G_LR,
        "G_OPT" : str(g_opt),
        "D_LR" : D_LR,
        "D_OPT" : str(d_opt),
        "SEED" : SEED
        }, f)

if SAVE_G_PTS:
    with open(os.path.join(RESULTS_DIR, G_POINT_SAVE_FILENAME), "wb") as f:
        pickle.dump(np.concatenate(G_POINT_REPR_PTS), f)

discriminator_reprs = np.array(discriminator_reprs)[...,0]
print("Dimensionality of discriminator representation:",discriminator_reprs.shape)
pca = PCA(n_components = 2)
pca.fit(discriminator_reprs)
ldr = pca.transform(discriminator_reprs)
# Low-dimensional Representation

if GENERATE_PLOTS:
    import seaborn as sns; sns.set(style="white", palette="muted", color_codes=True)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.set_title("Estimate of V(D,G)")
    sns.lineplot(np.repeat(training_logs["iteration"], MB_SIZE), np.concatenate(training_logs["v"]), ax=ax1)
    ax2.set_title("Discriminator Accuracy")
    sns.lineplot(np.repeat(training_logs["iteration"], MB_SIZE), np.concatenate(training_logs["d_real_acc"]), label="Real Data", ax=ax2)
    sns.lineplot(np.repeat(training_logs["iteration"], MB_SIZE), np.concatenate(training_logs["d_fake_acc"]), label="Fake Data", ax=ax2)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax3.set_title("Gradient Magnitude")
    sns.lineplot(np.repeat(training_logs["iteration"], NUM_MBS), np.concatenate(training_logs["d_grad_norm"]), legend=False, label="Discriminator Gradient", ax=ax3)
    ax4 = ax3.twinx()
    sns.lineplot(np.repeat(training_logs["iteration"], NUM_MBS), np.concatenate(training_logs["g_grad_norm"]), legend=False, label="Generator Gradient", color="orange", ax=ax4)
    ax4.legend(ax3.get_lines() + ax4.get_lines(), [line.get_label() for line in ax3.get_lines()] + [line.get_label() for line in ax4.get_lines()])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    n = ldr.shape[0]
    cmap = plt.get_cmap("Reds")
    for i in range(n):
        plt.scatter(ldr[i,0],ldr[i,1],color=cmap(i/n))
        if i < n - 1:
            plt.plot(ldr[i:i+2,0], ldr[i:i+2,1], color=cmap((i+0.5)/n))
    plt.savefig(os.path.join(RESULTS_DIR, "pca.png"))
    if SHOW_PLOTS:
        plt.show()