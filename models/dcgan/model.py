import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.utils import save_image

from models.dcgan.config import Config
from models.dcgan.discriminator import Discriminator
from models.dcgan.generator import Generator

config = Config()

os.makedirs("images", exist_ok=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def make_data_loader(n_samples: int, class_to_train: int = None) -> DataLoader:
    dataset = provide_dataset(class_to_train)
    train_indices = np.random.choice(dataset.data.shape[0], n_samples, replace=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        # shuffle=True,
        sampler=SubsetRandomSampler(train_indices)
    )
    return dataloader


def provide_dataset(class_to_train: int = None) -> datasets:
    data_path: str = "/data/cifar"

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

    return dataset


def train(n_samples: int, class_to_train: int = None):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if config.cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    dataloader = make_data_loader(n_samples, class_to_train)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    Tensor = torch.cuda.FloatTensor if config.cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(config.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], config.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] - [N samples: %d]"
                % (epoch, config.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), n_samples)
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % (round(n_samples/config.batch_size)) == 0:
                save_image(gen_imgs.data[:25],
                           f"images/dcgan/cifar/cifar10_class_{class_to_train}_samples_{n_samples}_{batches_done}.png",
                           nrow=5, normalize=True)
    torch.save(generator, f"models/saved_models/dcgan/cifar10_class_{class_to_train}_samples_{n_samples}.pt")
