import torch
import torchvision

train = torchvision.datasets.MNIST('mnist_data', train=True, download=True)
