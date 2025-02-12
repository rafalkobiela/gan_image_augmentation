import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from data_provider.create_dataset import create_dataset
from models.dcgan.model import train as train_dc_gan
from models.dcgan.config import Config
from testing_models.simple_conv import train_and_test
from sklearn.metrics import f1_score


def generate_image():
    config = Config()
    Tensor = torch.cuda.FloatTensor if config.cuda else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (32, config.latent_dim))))
    model = load_model()
    gen_imgs = model(z)
    save_image(gen_imgs[:4], "images/dcgan/gen/test3.png", nrow=2, normalize=True)
    # print(gen_imgs.cpu().detach().numpy())


def load_model() -> nn.Module:
    PATH = "models/saved_models/dcgan/cifar10_class_0_samples_5000.pt"
    model = torch.load(PATH, map_location=torch.device('cpu'))
    model.eval()
    return model


def train_imbalanced_gans():
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
    #  'truck': 9} - cifar-10
    class_to_train = 0  #
    numbers_of_samples = np.r_[5000:10:-100]
    for number_of_samples in numbers_of_samples:
        train_dc_gan(number_of_samples, class_to_train)


def train_discriminator(true_samples: int, verbose: int = 0, gan: bool = True):
    # X, y, X_test, y_test = create_dataset(true_samples)
    # tensor_x = torch.Tensor(X)
    # tensor_y = torch.Tensor(y)
    # my_dataset = TensorDataset(tensor_x, tensor_y)
    # my_dataloader = DataLoader(my_dataset)
    y_true, y_pred = train_and_test(true_samples, verbose, gan)
    score = f1_score(y_true, y_pred)
    if verbose:
        print(score)
    return score


if __name__ == "__main__":
    # generate_image()
    # train_imbalanced_gans()

    # train_discriminator(3000, verbose=1, gan=False)

    scores_dict = {}
    for true_samples in tqdm(np.r_[2500:5001:100]):

        scores = []
        for _ in range(10):
            score = train_discriminator(2500, 0, False)
            scores.append(score)
        scores_dict[true_samples] = np.mean(score)
    print(scores_dict)

