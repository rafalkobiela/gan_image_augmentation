import os

from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torchvision.utils import save_image
from sklearn.utils import shuffle

from models.dcgan.config import Config
from data_provider.generate_fakes import gan_generate_image, crops_generate_image
import numpy as np
from imgaug import augmenters as iaa
import tensorflow



def provide_dataset(class_to_train: int = None) -> np.array:
    data_path: str = "/home/rkobiela/projects/gan_image_augmentation/data/cifar"
    config = Config()
    os.makedirs(data_path, exist_ok=True)

    dataset = datasets.CIFAR10(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(config.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    if class_to_train is not None:
        dataset.targets = torch.tensor(dataset.targets)
        idx = dataset.targets == class_to_train
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx.numpy().astype(np.bool)]

    return dataset.data


def provide_test_dataset(class_to_train1: int = None, class_to_train2: int = None):
    data_path: str = "/home/rkobiela/projects/gan_image_augmentation/data/cifar"
    config = Config()
    os.makedirs(data_path, exist_ok=True)

    dataset = datasets.CIFAR10(
        data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(config.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    if class_to_train1 is not None:
        dataset.targets = torch.tensor(dataset.targets)
        idx = (dataset.targets == class_to_train1) | (dataset.targets == class_to_train2)
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx.numpy().astype(np.bool)]

    data = dataset.data
    return np.transpose(data, (0, 3, 1, 2)), dataset.targets.detach().numpy()


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    res = m * array + b
    return res


def create_dataset(leave_n_true_samples: int, gan: bool = True):
    dataset_1 = provide_dataset(1).astype(float)
    dataset_0 = provide_dataset(0).astype(float)
    dataset_0 = np.transpose(dataset_0, (0, 3, 1, 2))
    dataset_1 = np.transpose(dataset_1, (0, 3, 1, 2))

    samples_to_leave = np.random.choice(dataset_0.shape[0], leave_n_true_samples, replace=False)
    dataset_0 = dataset_0[samples_to_leave]

    fakes_to_generate = dataset_1.shape[0] - leave_n_true_samples
    if gan:
        fakes = gan_generate_image(fakes_to_generate)
        fakes = fakes.cpu().detach().numpy()
        fakes_scaled = rescale_linear(fakes, 0, 255)
    else:
        fakes = crops_generate_image(dataset_0, fakes_to_generate)
        fakes_scaled = fakes


    # fakes2
    # fakes = np.transpose(fakes, (0, 2,3,1))

    dataset_0 = np.concatenate([fakes_scaled, dataset_0], axis=0)
    # dataset_0.shape
    # dataset_1.shape
    X = np.concatenate([dataset_0, dataset_1], axis=0)
    y = np.concatenate([np.array([0] * dataset_0.shape[0]), np.array([1] * dataset_1.shape[0])], axis=0)
    X, y = shuffle(X, y)

    X_test, y_test = provide_test_dataset(0, 1)

    return X, y, X_test, y_test
