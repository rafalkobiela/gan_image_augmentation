import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

from data_provider.create_dataset import create_dataset


def train_and_test(true_samples, verbose = 1, gan: bool = True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 5
    num_classes = 2
    batch_size = 100
    learning_rate = 0.001

    X, y, X_test, y_test = create_dataset(true_samples, gan)
    X_train = torch.Tensor(X)
    y_train = torch.Tensor(y)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # # MNIST dataset
    # train_dataset = torchvision.datasets.MNIST(root='data/mnist',
    #                                            train=True,
    #                                            transform=transforms.ToTensor(),
    #                                            download=True)
    #
    # test_dataset = torchvision.datasets.MNIST(root='data/mnist',
    #                                           train=False,
    #                                           transform=transforms.ToTensor())
    #
    # # Data loader
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=True)

    # Convolutional neural network (two convolutional layers)
    class ConvNet(nn.Module):
        def __init__(self, num_classes=num_classes):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(2048, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = int(len(train_loader.dataset) / batch_size)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).view(-1).long()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0 and verbose:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print('Test Accuracy of the model on the 2000 test images: {} %'.format(100 * correct / total))

    return labels.cpu().detach().numpy(), predicted.cpu().detach().numpy()
