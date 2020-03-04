import torch
import numpy as np
from PIL import Image

def get_gradient_norm(parameters):
    """Given a parameter set, returns the 2-norm of the gradients
    of all the parameters concatenated together. Note that this is
    NOT comparable among different architectures, but this appears
    to be the standard.

    Arguments:
    parameters -- set of parameters to find norm of; can be obtained
                  from a model with model.parameters()
    """
    grads = []
    for param in parameters:
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return torch.norm(grads, p=2)


def sample_latent_prior(latent_dim, batch_size):
    """Samples from latent space; the prior distribution on noise.
    For now, this distribution is assumed to be the multivariate
    standard normal distribution, N(0,1)^d, where d is latent_dim.

    Arguments:
    latent_dim -- the dimension of the latent space to sample from
    batch_size -- the number of samples to take, for use as a batch
    """
    z = np.random.normal(size=(batch_size, latent_dim))
    return torch.from_numpy(z).float()


def sample_generator(generator, latent_dim, batch_size):
    """Samples from a generator using the default latent prior
    (multivariate normal distribution), and the given batch size.
    This does NOT preserve gradients, so only use for inference, not
    training.

    Arguments:
    generator -- the generator model to sample from
    latent_dim -- the dimension of latent space; must match generator
    batch_size -- the number of samples to take, for use as a batch
    """
    z = sample_latent_prior(latent_dim, batch_size)
    with torch.no_grad():
        fake = generator(z)
    return fake.numpy()

def convert_to_image(array):
    """Turns a numpy array (normalized from 0 to 1) into an image
    by renormalizing to 0 to 255, clipping, and converting to a PIL
    Image object
    """
    return Image.fromarray(np.clip(255 * array, 0, 255).astype(np.uint8))

def tile_images(images, tile_shape, upscale=1):
    """Tiles the given images into the desired shape.

    Arguments:
    images -- list of numpy arrays (images) to tile, the list must be
              nonempty, and the images must be the same shape; it must
              also contain enough images to fill out the tiling
    tile_shape -- 2-tuple, the shape to tile the images into
    upscale -- (optional) factor to scale images up by (linear interp)
    """
    img_shape = images[0].shape

    assert 2 <= len(img_shape) <= 3

    if len(img_shape) == 2:
        tiled_shape = (img_shape[0] * tile_shape[0],
                        img_shape[1] * tile_shape[1])
    else:
        tiled_shape = (img_shape[0] * tile_shape[0],
                        img_shape[1] * tile_shape[1],
                        img_shape[2])

    tiled = np.zeros(shape=tiled_shape, dtype=images[0].dtype)

    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            img_index = i * tile_shape[1] + j
            img = images[img_index]
            tiled[i * img_shape[0] : (i + 1) * img_shape[0],
                  j * img_shape[1] : (j + 1) * img_shape[1]] = img

    return tiled.repeat(upscale, axis=0).repeat(upscale, axis=1)