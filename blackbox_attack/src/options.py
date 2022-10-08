import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default = 10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default = 0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--epochs', type=int, default = 100,
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default = 10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default = 50,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default = 0.1,
                        help='learning rate')
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='if 0, then all clients have non-overlapping data; \
                        otherwise, in equal case, each client has dataset size = overlap ')
    parser.add_argument('--file_name', type=str, default='',
                        help='file name.')
    parser.add_argument('--seed', type=int, default = 2022,
                        help="random seed")
    parser.add_argument('--solver', type=str, default='FedZO', help="type \
                        of solver")
    parser.add_argument('--balance', type=float, default=1.0, help="balance \
                        parameter of attack loss & distortion loss")
    parser.add_argument('--target_label', type=int, default=4, help="the target \
                        to attack class")
    ############################### FedZO hyperparameter ###############################
    parser.add_argument('--num_dir', type=int, default=5, help="number of directions  \
                       in the zeroth-order gradient estimator")
    parser.add_argument('--step_size', type=float, default=0.001, help="step size for  \
                       zero order estimator")
    
    ####################################################################################

    ############################### AirComp hyperparameter ###############################
    parser.add_argument('--overAir', type=int, default=0, help="apply AirComp aggregation")
    parser.add_argument('--SNR', type=float, default=0., help="signal-noise ratio")
    
    ####################################################################################
                        
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    args = parser.parse_args()
    return args
