'''
Implementation of ZONE-S algorithm, proposed in 
<Hajinezhad, Davood, Mingyi Hong, and Alfredo Garcia. "ZONE: Zeroth-order nonconvex multiagent optimization over networks." IEEE Transactions on Automatic Control 64.10 (2019): 3995-4010.>
'''

from ast import Del
from ObjFunc import ObjectiveFunc
import copy
from utils import data_sampling, load_target_dataset, test_Attack_Accuracy
from options import args_parser
from models import CIFAR10, MNIST
import torch
from load_file import get_dataset, load_model
import numpy as np
from tqdm import tqdm
import math
import pickle

class ZONES_client():
    def __init__(self, args, train_dataset, ObjectiveFunc, model, DeltaImage):
        self.args = copy.deepcopy(args)
        self.train_dataset = train_dataset
        self.ObjectiveFunc = ObjectiveFunc
        self.GlobalModel = model

        self.local_DeltaImage = copy.deepcopy(DeltaImage) # z in ZONES, i.e. local copy of global variable
        self.dual_DeltaImage_curr = copy.deepcopy(DeltaImage)  # \lambda in ZONES, i.e. dual variable variable
        self.dual_DeltaImage_prev = copy.deepcopy(DeltaImage) 

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.args.local_bs, shuffle=True)

        # this coefficient == 1 if equal distribution; otherwise = N * |D_i| / |D|
        self.args.coefficient = 1.0
        #self.args.coefficient = self.args.num_users * len(self.train_dataset) / float(self.args.totalCount) 

        self.alpha = 1.0 / self.args.num_users 
        self.rho = 1.0
    

    def client_update(self, DeltaImage, GlobalRound):
        '''
        
        the client performs local update once received DeltaImage from server
        - local epoch = self.args.local_ep
        - batch size = self.args.local_bs
        - # of local updates = local epoch
        input:
            - DeltaImage: the global DeltaImage
        
        output:
            the difference between updated DeltaImage and the original input
        '''
        self.dual_DeltaImage_prev = copy.deepcopy(self.dual_DeltaImage_curr)

        self.rho = np.sqrt(GlobalRound+1)
        self.rho = 1.0 / self.args.lr

        DeltaImage_copy = copy.deepcopy(DeltaImage)

        overall_Loss, attack_Loss, distortion_Loss, attack_success, attack_total = None, None, None, None, None

        G_Func = torch.zeros_like(DeltaImage_copy).to(self.args.device)
        
        for H in range(self.args.local_ep):
            images_id = np.random.choice([i for i in range(len(self.train_dataset))], self.args.local_bs, replace=False).tolist()
            images = torch.stack([self.train_dataset[i][0] for i in images_id], dim = 0).to(self.args.device)

            grad, _, _ = self.ObjectiveFunc.GradEst(images, DeltaImage_copy, self.args.step_size, form='gaussian')
            G_Func += grad
        G_Func /= self.args.local_ep
        G_Func *= self.args.coefficient
        
        self.local_DeltaImage = DeltaImage_copy - 1.0 / (self.alpha * self.rho) *(self.dual_DeltaImage_curr + G_Func)
        self.dual_DeltaImage_curr = self.dual_DeltaImage_curr + (self.alpha * self.rho) * (self.local_DeltaImage - DeltaImage_copy)
        
        #overall_Loss, attack_Loss, distortion_Loss = self.client_checkLoss(DeltaImage_update)

        #attack_success, attack_total = self.client_checkAcc(DeltaImage_update)      


        return None
    
    def client_remain(self, DeltaImage, GlobalRound):
        self.rho = np.sqrt(GlobalRound+1)
        self.rho = 1.0 / self.args.lr

        self.dual_DeltaImage_prev = copy.deepcopy(self.dual_DeltaImage_curr)
        self.local_DeltaImage = copy.deepcopy(DeltaImage)


    def client_checkLoss(self, DeltaImage):
        # check loss
        overall_Loss_list, attack_Loss_list, distortion_Loss_list = [], [], []
        for images, labels in self.train_loader:
            ave_overall_Loss, ave_attack_Loss, ave_distortion_Loss = self.ObjectiveFunc.Loss_Func(images, DeltaImage)
            
            overall_Loss_list.append(ave_overall_Loss)
            attack_Loss_list.append(ave_attack_Loss)
            distortion_Loss_list.append(ave_distortion_Loss)
        overall_Loss = sum(overall_Loss_list)/len(overall_Loss_list)
        attack_Loss = sum(attack_Loss_list)/len(attack_Loss_list)
        distortion_Loss = sum(distortion_Loss_list)/len(distortion_Loss_list)

        return overall_Loss, attack_Loss, distortion_Loss
    
    def client_checkAcc(self, DeltaImage):
        # check accuracy
        attack_success, attack_total = test_Attack_Accuracy(self.train_loader, DeltaImage, self.GlobalModel, self.args, self.args.target_label)
        return attack_success, attack_total
            
        


class ZONES_server():
    def __init__(self, args, model, train_dataset, test_dataset, DeltaImage = None):
        self.args = args
        self.GlobalModel = model
        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=256, shuffle=False)

        self.test_dataset = test_dataset
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=256, shuffle=False)

        if DeltaImage == None:
            self.GlobalDeltaImage = torch.zeros_like(self.train_dataset[0][0]).to(self.args.device)
        else:
            self.GlobalDeltaImage = copy.deepcopy(DeltaImage)
        
        self.GlobalObjective = ObjectiveFunc(self.GlobalModel, self.args)
        self.userdataDict = data_sampling(self.train_dataset, self.args)

        # count the total number of data samples (include overlapping)
        totalCount = 0
        for i in range(self.args.num_users):
            totalCount += len(self.userdataDict[i])
        self.args.totalCount = totalCount

        self.ClientDict = {}

        for i in range(self.args.num_users):
            local_dataset = torch.utils.data.Subset(self.train_dataset, self.userdataDict[i])
            self.ClientDict[i] = ZONES_client(self.args, local_dataset, self.GlobalObjective, self.GlobalModel, self.GlobalDeltaImage)
    
    def SelectClients(self):
        '''
        ZONES algorithms activates only one client in each round
        '''
        # randomly select
        m = 1
        Active_Clients = np.random.choice([i for i in range(self.args.num_users)], m, replace=False).tolist()

        return Active_Clients
    
    def CheckClientStatus(self, clientSet=None):
        '''
        Check loss for clients
        '''
        if clientSet == None:
            clientSet = [i for i in range(self.args.num_users)]
        overall_Loss_epoch, attack_Loss_epoch, distortion_Loss_epoch, attack_success_epoch, attack_total_epoch = [], [], [], [], []
        for i in clientSet:
            overall_Loss, attack_Loss, distortion_Loss = self.ClientDict[i].client_checkLoss(self.GlobalDeltaImage)
            #attack_success, attack_total = self.ClientDict[i].client_checkAcc(self.GlobalDeltaImage)
            
            overall_Loss_epoch.append(overall_Loss)
            attack_Loss_epoch.append(attack_Loss)
            distortion_Loss_epoch.append(distortion_Loss)
            #attack_success_epoch.append(attack_success)
            #attack_total_epoch.append(attack_total)
        
        overall_Loss_ave = sum(overall_Loss_epoch)/len(overall_Loss_epoch)
        attack_Loss_ave = sum(attack_Loss_epoch)/len(attack_Loss_epoch)
        distortion_Loss_ave = sum(distortion_Loss_epoch)/len(distortion_Loss_epoch)
        #attack_success_sum = sum(attack_success_epoch)
        #attack_total_sum = sum(attack_total_epoch)

        return overall_Loss_ave, attack_Loss_ave, distortion_Loss_ave
    
    def CheckDatasetAcc(self, dataloader):
        '''
        check attack accuracy
        '''
        attack_success, attack_total = test_Attack_Accuracy(dataloader, self.GlobalDeltaImage, self.GlobalModel, self.args, self.args.target_label)
        return attack_success, attack_total

    def ServerExcute(self):

        overall_Loss_all, attack_Loss_all, distortion_Loss_all, train_attack_success_all = [], [], [], []
        test_attack_success_all = []
        
        overall_Loss_ave, attack_Loss_ave, distortion_Loss_ave = self.CheckClientStatus()
        train_attack_success, train_attack_total = self.CheckDatasetAcc(self.train_loader)
        print("Initial Stage  ||   attack Loss : {:.6f}  ||   distortion Loss : {:.6f}  ||  overall Loss : {:.6f}".format(attack_Loss_ave, distortion_Loss_ave, overall_Loss_ave))
        print("Initial Stage  || train set --  Attack Success: {} / {} = {:.2f}% ".format(train_attack_success, train_attack_total, 100.0 * float(train_attack_success) / train_attack_total ))
        
        test_attack_success, test_attack_total = self.CheckDatasetAcc(self.test_loader)
        print("Initial Stage  || test set  --  Attack Success: {} / {} = {:.2f}% \n".format(test_attack_success, test_attack_total, 100.0 * float(test_attack_success) / test_attack_total ))

        overall_Loss_all.append(overall_Loss_ave)
        attack_Loss_all.append(attack_Loss_ave)
        distortion_Loss_all.append(distortion_Loss_ave)
        train_attack_success_all.append(100.0 * float(train_attack_success) / train_attack_total )
        test_attack_success_all.append(100.0 * float(test_attack_success) / test_attack_total)
        

        for E in tqdm(range(self.args.epochs)):
            
            '''
            sampling clients
            '''
            Active_Clients = self.SelectClients()


            '''
            clients perform updates
            '''
            upper_part = torch.zeros_like(self.GlobalDeltaImage)
            lower_part = 0.0
            
            for i in range(self.args.num_users):
                if i in Active_Clients:
                    self.ClientDict[i].client_update(self.GlobalDeltaImage, E)
                else:
                    self.ClientDict[i].client_remain(self.GlobalDeltaImage, E)
                
                upper_part += self.ClientDict[i].rho * self.ClientDict[i].local_DeltaImage + self.ClientDict[i].dual_DeltaImage_prev
                lower_part += self.ClientDict[i].rho
            
            '''
            update global variable
            '''
            self.GlobalDeltaImage = upper_part / lower_part


            '''
            train loss & train acc
            '''
            overall_Loss_ave, attack_Loss_ave, distortion_Loss_ave = self.CheckClientStatus()
            overall_Loss_all.append(overall_Loss_ave)
            attack_Loss_all.append(attack_Loss_ave)
            distortion_Loss_all.append(distortion_Loss_ave)

            train_attack_success, train_attack_total = self.CheckDatasetAcc(self.train_loader)
            train_attack_success_all.append(100.0 * float(train_attack_success) / train_attack_total )
            print("Comm Round {} ||   attack Loss : {:.6f}  ||   distortion Loss : {:.6f}  ||  overall Loss : {:.6f}".format(E, attack_Loss_ave, distortion_Loss_ave, overall_Loss_ave))
            print("Comm Round {} || train set --   Attack Success: {} / {} = {:.2f}% ".format(E, train_attack_success, train_attack_total, 100.0 * float(train_attack_success) / train_attack_total ))

            '''
            test acc
            '''
            test_attack_success, test_attack_total = self.CheckDatasetAcc(self.test_loader)
            test_attack_success_all.append(100.0 * float(test_attack_success) / test_attack_total)
            print("Comm Round {} || test set --   Attack Success: {} / {} = {:.2f}% \n".format(E, test_attack_success, test_attack_total, 100.0 * float(test_attack_success) / test_attack_total ))
            
        
        
        
        file_name = '../save/{}.pkl'.format(self.args.file_name)

        with open(file_name, 'wb') as f:
            pickle.dump([overall_Loss_all, attack_Loss_all, distortion_Loss_all, train_attack_success_all, test_attack_success_all], f)

        return self.GlobalDeltaImage
    
    def solve(self):
        return self.ServerExcute()




        

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
    target_train_dataset = load_target_dataset(train_dataset, GloablModel, args, target_label=args.target_label)
    target_test_dataset = load_target_dataset(test_dataset, GloablModel, args, target_label=args.target_label)
    MyServer = ZONES_server(args=args, model=GloablModel, train_dataset=target_train_dataset, test_dataset=target_test_dataset)

    MyServer.ServerExcute()


    