import copy
from load_file import get_dataset, load_dataset, load_model
from models import CIFAR10, MNIST
from options import args_parser

import os
import numpy as np
import random
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_target_dataset(dataset, model, args, target_label = 4):
    '''
    load target label (e.g. label = 4) dataset, i.e. the dataset contains only
    samples that belongs to target label & is correctly predicted by the model

    input:
        - dataset     : the complete dataset
        - model       : the well-trained model
        - args        : arguments
        - target_label: the target to-attack label

    output: 
        the dataset that contains only correctly predicted target label samples
    
    '''
    print('Start to get target dataset!\n')
    target_id_List = []
    for i in range(len(dataset)):
        if dataset[i][1] == target_label:
            target_id_List.append(i)

    target_dataset = torch.utils.data.Subset(dataset, target_id_List)
    
    
    target_dataloader, _ = load_dataset(train_dataset=target_dataset)

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    predict_label_list = []

    for images, labels in target_dataloader:
        if torch.cuda.is_available():
            images, labels = images.to(args.device), labels.to(args.device)
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predict_label_list.append(predicted)
    predict_label_list = torch.cat(predict_label_list)

    target_true_id_List = []
    for i in range(torch.numel(predict_label_list)):
        if int(predict_label_list[i]) == int(target_label):
            target_true_id_List.append(i)
    target_true_dataset = torch.utils.data.Subset(target_dataset, target_true_id_List)

    print('Target label {} has data size: {} / {} \n'.format(target_label, len(target_true_dataset), len(target_dataset)))

    print('Already get target dataset!\n')
    
    return target_true_dataset
    
def test_Attack_Accuracy(data_loader, DeltaImage, model, args, target_label):
    '''
    test the attack success rate of the delta image on the given data loader

    input:
        - data_loader : the to-be tested dataset
        - DeltaImage  : the purturbation image
        - model       : the well-trained model
        - args        : arguments
        - target_label: the target to-attack label

    output: 
        - count_success   : # of successfully attacked images
        - dataLoader_size : data set size
    
    '''
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    predict_label_list = []
    for images, labels in data_loader:
        if torch.cuda.is_available():
            images, labels = images.to(args.device), labels.to(args.device)
        TempImages = Tanh_Transform(images_batch=images, Delta_image=DeltaImage)
        TempImages = Variable(TempImages)
        outputs = model(TempImages)
        _, predicted = torch.max(outputs.data, 1)
        predict_label_list.append(predicted)
    predict_label_list = torch.cat(predict_label_list)
    dataLoader_size = torch.numel(predict_label_list)
    

    count_success = 0
    for i in range(dataLoader_size):
        if int(predict_label_list[i]) != int(target_label):
            count_success += 1

    #print('Attack Success: {} / {} = {:.2f}% \n'.format(count_success, dataLoader_size, 100.0 * float(count_success)/dataLoader_size ))
    return count_success, dataLoader_size

def image_label_prediction(images, model, args):
    '''
    get the prediction labels of the given images

    input:
        - images : batches of images
        - model  : the well-trained model
        - args   : arguments
    
    output:
        predicted labels for images
    '''
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    if torch.cuda.is_available():
        images = images.to(args.device)
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

@torch.no_grad()
def Tanh_Transform(images_batch, Delta_image):
    '''
    combine images batch with the adversarial delta image in tanh space to ensure it still lies within [-0.5, 0.5]^d dimension

    input: 
        - images_batch: orginal images batch, e.g. for cifar-10, it has shape [batch_size, 3, 32, 32]
        - Delta_image: the adversarial delta image, e.g. for cifar-10, it has shape [3, 32, 32]

    output:
        tanh-transformed images batch
    '''
    
    Delta_image_tiled = torch.tile(torch.unsqueeze(Delta_image, 0), (images_batch.shape[0], 1, 1, 1))
    #print(Delta_image_tiled)
    converted_images_batch = torch.atanh(2.0 * images_batch) + Delta_image_tiled
    #print(converted_images_batch)
    inverse_images_batch = 0.5 * torch.tanh(converted_images_batch)
    
    return inverse_images_batch

def data_equal_sampling(dataset, args):
    '''
    all clients have same amount of local data
    '''
    user_data_dict = {}
    all_idxs = [i for i in range(len(dataset))]

    if args.overlap == 0:
        num_items = int(len(dataset)/args.num_users)
        for i in range(args.num_users):
            user_data_dict[i] = np.random.choice(all_idxs, num_items,
                                                replace=False).tolist()
            all_idxs = list(set(all_idxs) - set(user_data_dict[i]))
    else:
        for i in range(args.num_users):
            user_data_dict[i] = np.random.choice(all_idxs, args.overlap,
                                                replace=False).tolist()
    return user_data_dict
            
def data_unequal_sampling(dataset, args):
    '''
    all clients have same amount of local data
    '''
    user_data_dict = {}
    all_idxs = [i for i in range(len(dataset))]

    if args.overlap == 0:
        # a special case of distribution, 1/5 100, 1/5 250, 1/5 500, 1/5 750, 1/5 900
        if args.num_users == 10:
            sampleList = [100, 100, 250, 250, 500, 500, 750, 750, 900, 900]
            for i in range(args.num_users):
                sampleNum = min(len(all_idxs), sampleList[i])
                user_data_dict[i] = np.random.choice(all_idxs, sampleNum, replace=False).tolist()
            all_idxs = list(set(all_idxs) - set(user_data_dict[i]))
        else:
            avgSample = round(len(all_idxs) / args.num_users) # [half of avg, double of avg]

            #print('avg sample', avgSample)

            all_agents = [i for i in range(args.num_users)]
            
            # in 5000 samples & 50 agents case, first assign 50 samples to each device 
            sampleList = int(avgSample/2) * np.ones(args.num_users, dtype=int)
            remain_budget = len(all_idxs) - np.sum(sampleList)

            remain_budget_copy = remain_budget

            # assign remaining samples to all agents, each agents receive [0, 150] samples uniformly
            while remain_budget_copy > 0:
                remain_budget_copy = remain_budget
                all_agents_copy = copy.deepcopy(all_agents)
                sampleList_copy = copy.deepcopy(sampleList)

                while remain_budget_copy > 0 and len(all_agents_copy) > 0:
                    pickedAgent = np.random.choice(all_agents_copy, 1, replace=False)

                    all_agents_copy = list(set(all_agents_copy) - set(pickedAgent))

                    assign_num = min(np.random.randint(0, 2 * avgSample - int(avgSample/2) + 1), remain_budget_copy)

                    remain_budget_copy -= assign_num

                    sampleList_copy[pickedAgent] += assign_num

                    #print('assign samples = {} to agent {}'.format(assign_num, pickedAgent))
            sampleList = sampleList_copy
            print('\nData Assignment:')
            print('min    : ', np.min(sampleList))
            print('max    : ', np.max(sampleList))
            print('median : ', np.median(sampleList))
            print('mean   : ', np.mean(sampleList))
            print('stddev : ', np.std(sampleList))
            

            for i in range(args.num_users):
                sampleNum = min(len(all_idxs), sampleList[i])
                user_data_dict[i] = np.random.choice(all_idxs, sampleNum, replace=False).tolist()
            all_idxs = list(set(all_idxs) - set(user_data_dict[i]))
        
    else:
        
        
        ave_samples = int(len(dataset)/args.num_users)
        for i in range(args.num_users):
            num_items = np.random.randint(int(ave_samples/2) + 1, min(max(args.overlap, int(ave_samples/2) + 2), int(len(dataset)/4)))
            user_data_dict[i] = np.random.choice(all_idxs, num_items,
                                                replace=False).tolist()
    return user_data_dict




def data_sampling(dataset, args):
    '''
    sampling dataset for each client in federated setting

    input:
        - dataset: the dataset 
        - args   : arguments
    
    output:

    '''
    print('data sampling ......')
    print('unequal: {}'.format(args.unequal))
    print('overlap: {}'.format(args.overlap))

    if args.unequal == 0:
        '''
        all clients have same amount of local data
        '''  
        user_data_dict = data_equal_sampling(dataset, args)
    else:
        '''
        clients have different amount of local data
        ''' 
        user_data_dict = data_unequal_sampling(dataset, args)

    print('data sampled!')

    return user_data_dict





def plot_image(args, image, idx):
    if not os.path.exists('../save/{}'.format(args.file_name)):
        os.makedirs('../save/{}'.format(args.file_name))
    TempImage = image + 0.5 * torch.ones_like(image)
    save_image(TempImage, '../save/{}/{}.png'.format(args.file_name, idx))

def plot_delta_image(args, DeltaImage, idx='delta', image = None):
    if image == None:
        image = torch.zeros_like(DeltaImage)
    
    TempImage = torch.unsqueeze(image, 0)
    TempImage = Tanh_Transform(TempImage, DeltaImage)
    TempImage = torch.squeeze(TempImage, 0)
    plot_image(args=args, image=TempImage, idx=idx)



if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.file_name = args.file_name.replace('.', '')
    
    '''
    MNISTmodel = MNIST().to(args.device)
    load_model(MNISTmodel, '../models/mnist.pt')
    MNIST_train_dataset, MNIST_test_dataset = get_dataset(dataset='mnist')
    for label in range(10):
        load_target_dataset(MNIST_train_dataset, MNISTmodel, args, target_label=label)
    '''

    '''
    MNISTmodel = MNIST().to(args.device)
    load_model(MNISTmodel, '../models/fmnist.pt')
    MNIST_train_dataset, MNIST_test_dataset = get_dataset(dataset='fmnist')
    for label in range(10):
        load_target_dataset(MNIST_train_dataset, MNISTmodel, args, target_label=label)
    '''

    CIFAR10model = CIFAR10().to(args.device)
    load_model(CIFAR10model, '../models/cifar10.pt')
    CIFAR10_train_dataset, CIFAR10_test_dataset = get_dataset(dataset='cifar10')
    for label in range(10):
        load_target_dataset(CIFAR10_train_dataset, CIFAR10model, args, target_label=label)