from torchvision import datasets as torchdatasets
import numpy as np

class MNIST:
    def __init__(self, normalization="sigmoid", shape="image", download_dir="../data"):
        """Instantiates a new MNIST dataset instance.

        Arguments:
        normalization -- one of "sigmoid" [0,1] or "tanh" [-1,1]
        shape -- one of "image" (28x28) or "flat" (784)
        download_dir -- where to download data to
        """
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

    def __str__(self):
        """Contains relevant dataset hyperparameters"""
        return f"<{self.__class__.__name__} object, {self.normalization}, {self.shape}>"