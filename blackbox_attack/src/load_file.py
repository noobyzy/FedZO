from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
from models import CIFAR10, MNIST
from options import args_parser

def get_dataset(dataset):
    
    if dataset == 'cifar10':
        data_dir = '../../dataset/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    elif dataset == 'mnist' or 'fmnist':
        if dataset == 'mnist':
            data_dir = '../../dataset/mnist/'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        else:
            data_dir = '../../dataset/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.,))])

            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    print('Already get dataset!\n')

    return train_dataset, test_dataset

def load_dataset(train_dataset=None, test_dataset=None, local_bs=64):
    """ 
    Load MNIST/CIFAR-10 dataset
    
    input: train_dataset, test_dataset
    
    output: minibatches of train and test sets 
    """
    
    train_loader = None
    test_loader = None
    # Data Loader (Input Pipeline)
    if train_dataset != None:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=local_bs, shuffle=False)
    if test_dataset != None:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def load_model(model, filename):
    """ 
    Load the trained model 
    """
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(filename).items()})

def test_model(model, test_loader, args):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.to(args.device), labels.to(args.device)
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy: {}/{} = {:.4f}%'.format( correct, total, 100.0 * correct / total))


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MNISTmodel = MNIST().to(args.device)
    CIFAR10model = CIFAR10().to(args.device)

    
    load_model(CIFAR10model, '../models/cifar10.pt')
    
    load_model(MNISTmodel, '../models/mnist.pt')

    MNIST_train_dataset, MNIST_test_dataset = get_dataset(dataset='mnist')

    CIFAR10_train_dataset, CIFAR10_test_dataset = get_dataset(dataset='cifar10')

    _, MNIST_test_loader = load_dataset(test_dataset=MNIST_test_dataset)

    _, CIFAR10_test_loader = load_dataset(test_dataset=CIFAR10_test_dataset)

    test_model(MNISTmodel, MNIST_test_loader, args)
    test_model(CIFAR10model, CIFAR10_test_loader, args)


