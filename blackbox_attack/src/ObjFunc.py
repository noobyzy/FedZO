from ast import Del
import copy
import torch
import torch.nn as nn
from load_file import get_dataset, load_model
from models import CIFAR10, MNIST
from options import args_parser
from utils import load_target_dataset, test_Attack_Accuracy, Tanh_Transform
from torch.autograd import Variable

class ObjectiveFunc():
    def __init__(self, model, args):
        self.model = copy.deepcopy(model)
        self.args = copy.deepcopy(args)
        
        # number of function evaluation
        self.funcEval = 0

        if self.args.dataset == 'mnist' or self.args.dataset == 'fmnist':
            self.dimension = 784
        elif self.args.dataset == 'cifar10':
            self.dimension = 3072

        self.model.eval()

    
    def RandVec_Gen(self, form='sphere'):
        '''
        randomly generate a vector uniformly sampled from the unit sphere, 
        the vector shares the same size as the one image sample

        output:
            a tensor with l2 norm = 1
        '''

        if form == 'sphere':
            # randomly sampled on the sphere
            RandVec = torch.randn(self.dimension).to(self.args.device)
            RandVec = RandVec / torch.linalg.norm(RandVec, ord = 2)
        elif form == 'gaussian':
            RandVec = torch.randn(self.dimension).to(self.args.device)

        if self.args.dataset == 'mnist' or self.args.dataset == 'fmnist':
            RandVec = torch.reshape(RandVec, (1, 28, 28))
        elif self.args.dataset == 'cifar10':
            RandVec = torch.reshape(RandVec, (3, 32, 32))

        return RandVec
    
    @torch.no_grad()
    def Loss_Func(self, images_batch, Delta_image):
        '''
        preprocess images
        '''
        images_batch, Delta_image = images_batch.to(self.args.device), Delta_image.to(self.args.device)
        Tanh_images_batch = Tanh_Transform(images_batch, Delta_image)
        Tanh_images_batch = Variable(Tanh_images_batch)
        
        '''
        calculate the attack loss part
        '''
        outputs = self.model(Tanh_images_batch)
        
        targetLabel_score = copy.deepcopy(outputs[:, self.args.target_label].detach())
        
        outputs[:, self.args.target_label] = -10000.0 * torch.ones_like(outputs[:, self.args.target_label])
        nontargetLabel_score = torch.max(outputs, dim=1).values
        
        attack_Loss = torch.maximum(targetLabel_score - nontargetLabel_score, torch.zeros_like(targetLabel_score - nontargetLabel_score))
        ave_attack_Loss = torch.mean(attack_Loss).item()


        '''
        calculate the distortion loss part
        '''
        distortion_Loss = torch.square(torch.norm((Tanh_images_batch - images_batch).reshape(images_batch.shape[0], -1), p='fro', dim=1))
        ave_distortion_Loss = torch.mean(distortion_Loss).item()
        
        
        ave_overall_Loss = ave_attack_Loss + self.args.balance * ave_distortion_Loss

        return  ave_overall_Loss, ave_attack_Loss, ave_distortion_Loss

    @torch.no_grad()
    def GradEst(self, images_batch, Delta_image, mu, form='sphere'):

        images_batch, Delta_image = images_batch.to(self.args.device), Delta_image.to(self.args.device)
        
        EstGrad = torch.zeros_like(Delta_image).to(self.args.device)

        currState_Func_eval, attack_Loss, distortion_Loss = self.Loss_Func(images_batch, Delta_image)

        coefficient = self.dimension / ( mu * self.args.num_dir)

        for _ in range(self.args.num_dir):
            RandomVec = self.RandVec_Gen(form=form).to(self.args.device)
            newState_Func_eval, _, _ = self.Loss_Func(images_batch, Delta_image + mu * RandomVec)
            EstGrad += coefficient * (newState_Func_eval - currState_Func_eval) * RandomVec
        
        return EstGrad, attack_Loss, distortion_Loss






if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = 'cifar10'

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
    
    #target_dataset = load_target_dataset(MNIST_train_dataset, MNISTmodel, args, target_label=4) # [batch_size, 1, 28, 28]
    target_train_dataset = load_target_dataset(train_dataset, GloablModel, args, target_label=args.target_label) # [batch_size, 3, 32, 32]

    train_loader = torch.utils.data.DataLoader(dataset=target_train_dataset, batch_size=64, shuffle=True)

    DeltaImage = torch.zeros_like(train_dataset[0][0]).to(args.device)
    #DeltaImage /= torch.norm(DeltaImage, 'fro')

    MyObjective = ObjectiveFunc(GloablModel, args)

    attack_Loss_list, distortion_Loss_list = [], []
    for images, labels in train_loader:
        _, attack_Loss, distortion_Loss = MyObjective.Loss_Func(images, DeltaImage)
        attack_Loss_list.append(attack_Loss), distortion_Loss_list.append(distortion_Loss)
    ave_attack_Loss, ave_distortion_Loss = sum(attack_Loss_list)/len(attack_Loss_list), sum(distortion_Loss_list)/len(distortion_Loss_list)
    print("Initial Stage   ||      attack Loss : {:.6f}    ||       distortion Loss : {:.6f}    ||       overall Loss : {:.6f}".format(ave_attack_Loss, ave_distortion_Loss, ave_attack_Loss + ave_distortion_Loss))

    for epoch in range(args.epochs):
        attack_Loss_list, distortion_Loss_list = [], []
        for images, labels in train_loader:
            grad, attack_Loss, distortion_Loss = MyObjective.GradEst(images, DeltaImage, args.step_size)

            DeltaImage -= 0.0002 * grad
            
            attack_Loss_list.append(attack_Loss), distortion_Loss_list.append(distortion_Loss)

        ave_attack_Loss, ave_distortion_Loss = sum(attack_Loss_list)/len(attack_Loss_list), sum(distortion_Loss_list)/len(distortion_Loss_list)
        print("Epoch {}/{}    ||      attack Loss : {:.6f}    ||       distortion Loss : {:.6f}    ||       overall Loss : {:.6f}".format(epoch, args.epochs, ave_attack_Loss, ave_distortion_Loss, ave_attack_Loss + ave_distortion_Loss))
        #temp_train_loader = torch.utils.data.DataLoader(dataset=target_train_dataset, batch_size=64, shuffle=False)
        test_Attack_Accuracy(train_loader, DeltaImage, GloablModel, args, args.target_label)
    
    print("======================Training Completed!==============================\n")

    target_test_dataset = load_target_dataset(test_dataset, GloablModel, args, target_label=args.target_label)
    test_loader = torch.utils.data.DataLoader(dataset=target_test_dataset, batch_size=64, shuffle=False)
    print("test set atttack success rate")
    test_Attack_Accuracy(test_loader, DeltaImage, GloablModel, args, args.target_label)

