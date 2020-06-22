import numpy as np
import torch
from keras_preprocessing.image import ImageDataGenerator
from torch import nn
from torch.autograd import Variable

from models.dcgan.config import Config


def gan_generate_image(number_of_images: int = 100):
    config = Config()
    Tensor = torch.cuda.FloatTensor if config.cuda else torch.FloatTensor
    if number_of_images > 1000:
        times = number_of_images // 1000
        rest = number_of_images % 1000

        z = Variable(Tensor(np.random.normal(0, 1, (1000, config.latent_dim))))
        model = load_model()
        gen_imgs = model(z)

        for i in range(times - 1):
            z = Variable(Tensor(np.random.normal(0, 1, (1000, config.latent_dim))))
            model = load_model()
            gen_imgs_new = model(z)
            gen_imgs = torch.cat((gen_imgs, gen_imgs_new), 0)

        if rest > 0:
            z = Variable(Tensor(np.random.normal(0, 1, (rest, config.latent_dim))))
            model = load_model()
            gen_imgs_new = model(z)
            gen_imgs = torch.cat((gen_imgs, gen_imgs_new), 0)
        return gen_imgs

    else:
        z = Variable(Tensor(np.random.normal(0, 1, (number_of_images, config.latent_dim))))
        model = load_model()
        gen_imgs = model(z)
        return gen_imgs
    # print(gen_imgs.cpu().detach().numpy())


def crops_generate_image(X: np.ndarray, number_of_images: int = 100):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(X)
    itr = datagen.flow(X, batch_size=number_of_images)
    X_new = itr.next()
    return X_new




def load_model() -> nn.Module:
    PATH = "models/saved_models/dcgan/cifar10_class_0_samples_5000.pt"
    model = torch.load(PATH)
    model.eval()
    return model
