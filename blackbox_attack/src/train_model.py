import torch
import torch.nn as nn
from torch.autograd import Variable
from models import CIFAR10, MNIST
from options import args_parser
from load_file import get_dataset, test_model, load_dataset

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001


def train_mnist(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.item()))

def train_cifar10(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        if epoch%10==0 and epoch!=0:
            lr = lr * 0.95
            momentum = momentum * 0.5
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.item()))

def train_and_save_MNIST(args):
    MNISTmodel = MNIST().to(args.device)
    MNIST_train_dataset, MNIST_test_dataset = get_dataset(dataset='mnist')
    MNIST_train_loader, MNIST_test_loader = load_dataset(train_dataset=MNIST_train_dataset, test_dataset=MNIST_test_dataset, local_bs=batch_size)

    train_mnist(MNISTmodel, MNIST_train_loader)
    test_model(MNISTmodel, MNIST_test_loader, args)

    torch.save(MNISTmodel.state_dict(), '../models/mnist.pt')

def train_and_save_FMNIST(args):
    MNISTmodel = MNIST().to(args.device)
    MNIST_train_dataset, MNIST_test_dataset = get_dataset(dataset='fmnist')
    MNIST_train_loader, MNIST_test_loader = load_dataset(train_dataset=MNIST_train_dataset, test_dataset=MNIST_test_dataset, local_bs=batch_size)

    train_mnist(MNISTmodel, MNIST_train_loader)
    test_model(MNISTmodel, MNIST_test_loader, args)

    torch.save(MNISTmodel.state_dict(), '../models/fmnist.pt')

def train_and_save_CIFAR10(args):
    CIFAR10model = CIFAR10().to(args.device)
    CIFAR10_train_dataset, CIFAR10_test_dataset = get_dataset(dataset='cifar10')
    CIFAR10_train_loader, CIFAR10_test_loader = load_dataset(train_dataset=CIFAR10_train_dataset, test_dataset=CIFAR10_test_dataset, local_bs=batch_size)

    train_cifar10(CIFAR10model, CIFAR10_train_loader)
    test_model(CIFAR10model, CIFAR10_test_loader, args)

    torch.save(CIFAR10model.state_dict(), '../models/cifar10.pt')

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train_and_save_CIFAR10(args)
    #train_and_save_MNIST(args)
    train_and_save_FMNIST(args)

    