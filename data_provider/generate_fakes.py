import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from models.dcgan.config import Config


def generate_image(number_of_images:int = 100):
    config = Config()
    Tensor = torch.cuda.FloatTensor if config.cuda else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (number_of_images, config.latent_dim))))
    model = load_model()
    gen_imgs = model(z)
    return gen_imgs
    # print(gen_imgs.cpu().detach().numpy())


def load_model() -> nn.Module:
    PATH = "models/saved_models/dcgan/cifar10_class_0_samples_5000.pt"
    model = torch.load(PATH, map_location=torch.device('cpu'))
    model.eval()
    return model