import torch
import numpy as np

from options import args_parser
from models import CIFAR10, MNIST
from load_file import get_dataset, load_model
from utils import load_target_dataset, setup_seed, image_label_prediction, plot_image, plot_delta_image, Tanh_Transform
from alg_FedZO import FedZO_server
from alg_ZONES import ZONES_server
from alg_DZOPA import DZOPA_controller


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.file_name = '{}_{}_'.format(args.solver, args.dataset) + args.file_name + '_{}'.format(args.seed)
    setup_seed(args.seed)

    if args.dataset == 'cifar10':
        GloablModel = CIFAR10().to(args.device)
        load_model(GloablModel, '../models/cifar10.pt')
        train_dataset, test_dataset = get_dataset(dataset='cifar10')     
    elif args.dataset =='mnist':
        GloablModel = MNIST().to(args.device)
        load_model(GloablModel, '../models/mnist.pt')
        train_dataset, test_dataset = get_dataset(dataset='mnist')
    elif args.dataset =='fmnist':
        GloablModel = MNIST().to(args.device)
        load_model(GloablModel, '../models/fmnist.pt')
        train_dataset, test_dataset = get_dataset(dataset='fmnist')
    else:
        raise FileNotFoundError
    
    
    target_train_dataset = load_target_dataset(train_dataset, GloablModel, args, target_label=args.target_label)
    target_test_dataset = load_target_dataset(test_dataset, GloablModel, args, target_label=args.target_label)

    
    if args.solver == 'FedZO':
        solver = FedZO_server(args=args, model=GloablModel, train_dataset=target_train_dataset, test_dataset=target_test_dataset)
    elif args.solver == 'ZONES':
        solver = ZONES_server(args=args, model=GloablModel, train_dataset=target_train_dataset, test_dataset=target_test_dataset)
    elif args.solver == 'DZOPA':
        solver = DZOPA_controller(args=args, model=GloablModel, train_dataset=target_train_dataset, test_dataset=target_test_dataset)
    else:
        raise NotImplementedError
    

    DeltaImage = solver.solve()
    

    '''
    randomly pick images and visualize the results
    '''
    picked_images_id = np.random.choice([i for i in range(len(target_train_dataset))], 10, replace=False).tolist()
    picked_images_id = [i for i in range(10)]
    picked_images_batch = torch.stack([target_train_dataset[i][0] for i in picked_images_id], dim = 0).to(args.device)
    

    picked_images_predict = image_label_prediction(Tanh_Transform(picked_images_batch, DeltaImage), GloablModel, args).tolist()

    for i in range(len(picked_images_id)):
        plot_image(args=args, image=picked_images_batch[i], idx='idx_{}'.format(picked_images_id[i]))
        plot_delta_image(args=args, DeltaImage=DeltaImage, idx='idx_{}_ori_{}_pred_{}'.format(picked_images_id[i], args.target_label, picked_images_predict[i]), image=picked_images_batch[i])
    plot_delta_image(args=args, DeltaImage=DeltaImage)

    
