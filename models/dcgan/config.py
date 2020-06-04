import torch


class Config:
    def __init__(self):
        self.n_epochs = 800
        self.batch_size = 64
        self.lr = 0.0002
        self.b1 = 0.5 #adam: decay of first order momentum of gradient
        self.b2 = 0.999 #adam: decay of first order momentum of gradient
        self.n_cpu = 8
        self.latent_dim = 100 #dimensionality of the latent space
        self.img_size = 32
        self.channels = 3
        self.sample_interval = 400
        self.cuda = True if torch.cuda.is_available() else False
