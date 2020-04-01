from torchvision import datasets as torchdatasets
import numpy as np
from abc import ABC, abstractmethod
from util import *
import matplotlib.pyplot as plt

class Dataset(ABC):
    @abstractmethod
    def __init__(self, dataset_type, preferred_sample_size):
        """Instantiates a new instance of the dataset.

        Arguments:
        dataset_type -- string representing name of the dataset
        preferred_sample_size -- size of the sample to use when sampling progress
                                and writing periodic samples
        """
        super().__init__()
        self.dataset_type = dataset_type
        self.preferred_sample_size = preferred_sample_size

    @abstractmethod
    def sample_train(self, batch_size):
        """Samples from the "training dataset". For all intents and purposes,
        this dataset object only represents a training set, but in the future
        it may support sampling from test sets as well.

        Arguments:
        batch_size -- size of the batch to sample
        """
        pass

    @abstractmethod
    def write_sample(self, generator_sample, file_path, consistent):
        """Writes the given generator sample to the given file path. This is to
        require datasets to specify their own output format. For instance, for a
        dataset of images this might involve tiling images and outputting as a png.

        If consistent is true, then the same generator_sample passed in will
        correspond to the same sample from the latent space for the lifetime of
        this dataset instance. If true, datasets that display a real sample from the
        dataset next to the generator sample should keep this consistent as well.

        Arguments:
        generator_sample -- the sample from the generator; same format as data from sample_train
        file_path -- the file path to save the sample, with no extension
        consistent -- boolean indicating whether the generator samples come from the same latent sample
        """
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"<{self.dataset_type} dataset object>"

class MNIST(Dataset):
    """MNIST Handwritten Digit Dataset"""

    def __init__(self, normalization="sigmoid", shape="image", download_dir="../data"):
        """Instantiates a new MNIST dataset instance.

        Arguments:
        normalization -- one of "sigmoid" [0,1] or "tanh" [-1,1]
        shape -- one of "image" (28x28) or "flat" (784)
        download_dir -- where to download data to
        """
        super().__init__("MNIST", 64)
        self.normalization = normalization
        self.shape = shape
        self.mnist_train = torchdatasets.MNIST(download_dir, train=True, download=True).data
        self.mnist_test = torchdatasets.MNIST(download_dir, train=False, download=True).data
        self.num_train = self.mnist_train.shape[0]
        self.num_test = self.mnist_test.shape[0]

    def sample_train(self, batch_size):
        """Samples from MNIST training data.

        Arguments:
        batch_size -- size of batch, number of images to sample
        """
        x = self.mnist_train[np.random.randint(0, self.num_train, size=batch_size)].float()

        if self.normalization == "sigmoid":
            x = x / 255.
        elif self.normalization == "tanh":
            x = (2. * (x / 255.)) - 1.

        if self.shape == "image":
            pass
        elif self.shape == "flat":
            x = x.reshape(-1, 784)

        return x

    def write_sample(self, generator_sample, file_path, consistent):
        """Writes a sample of generator images"""
        sampled_ims = generator_sample.reshape(-1, 28, 28)
        im = convert_to_image(tile_images(sampled_ims, (8, 8), 4))
        im.save(file_path + ".png")

    def __str__(self):
        """Contains relevant dataset hyperparameters"""
        return f"<{self.__class__.__name__} object, {self.normalization}, {self.shape}>"


def smart_mean_selection(n_dims, n_clusters, iters=3, resolution=200):
    """Selects means that are approximately equidistant from each other, by voronoization"""
    def voronoize(means):
        """Approximates a voronoi algorithm by sampling points and adjusting means"""
        points = np.moveaxis(
            np.array(np.meshgrid(*[np.linspace(0, 1, resolution) for _ in range(n_dims)], indexing='ij')),
            0, -1).reshape(-1, means.shape[1])
        distances = ((points[:, None] - means[None])**2).sum(axis=-1)
        min_dists = distances.argmin(axis=1)
        return np.array([
                np.mean(points[min_dists == i], axis=0)
            for i in range(means.shape[0])])
    means = np.random.random((n_clusters, n_dims))
    for _ in range(iters):
        means = voronoize(means)
    return means



class GMM(Dataset):
    """Gaussian Mixture Model Toy Dataset"""

    def __init__(self, n_dims=2, n_clusters=2, equal_clusters=False):
        """Initializes a new GMM dataset instance.

        Arguments:
        n_dims -- number of dimensions of the GMM
        n_clusters -- number of distinct clusters to sample from
        equal_clusters -- whether to sample equally from the clusters (if False,
                        then distribution across clusters is randomly selected
                        from {n_clusters}-dimensional simplex)
        """
        super().__init__("GMM", 256)
        self.cluster_means = smart_mean_selection(n_dims, n_clusters)
        self.cluster_var = 0.05 + 0.05 * np.random.random((n_clusters, n_dims)) 
        self.cluster_dist = np.random.exponential(1, (n_clusters,))
        self.cluster_dist = self.cluster_dist / self.cluster_dist.sum()
        self.cluster_cdf = np.cumsum(self.cluster_dist)
        self.consistent_sample = None

    def sample_train(self, batch_size):
        """Samples from GMM.

        Arguments:
        batch_size -- size of batch, number of points to sample
        """
        batch = []
        for _ in range(batch_size):
            cluster = np.random.random()
            cluster = np.where(cluster < self.cluster_cdf)[0].min() # sampling from cdf
            batch.append(np.clip(np.random.normal(
                self.cluster_means[cluster,:],
                self.cluster_var[cluster,:]), 0, 1))
        return torch.from_numpy(np.array(batch)).float()

    def write_sample(self, generator_sample, file_path, consistent):
        """Writes a sample of generator points"""
        if self.consistent_sample is None: # and consistent
            self.consistent_sample = self.sample_train(self.preferred_sample_size)

        plt.close()
        plt.scatter(generator_sample[:,0], generator_sample[:,1], label="Fake")
        if consistent:
            plt.scatter(self.consistent_sample[:,0], self.consistent_sample[:,1], label="Real")
        else:
            sample = self.sample_train(self.preferred_sample_size)
            plt.scatter(sample[:,0], sample[:,1], label="Real")

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(file_path + ".png")
