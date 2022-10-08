import argparse


def argparse_option():
    parser = argparse.ArgumentParser('Arguments for OPP-PersonReID')
    parser.add_argument('--dream_person', type=int, default=1, help='num of person for dreaming')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')

    # optimization
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='Market-1501',
                        choices=['Market-1501', 'DukeMTMC-ReID', 'MARS', 'MSMT17'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='~/codes/OPP-PersonReID/Market-1501/pytorch/',
                        choices=['~/codes/OPP-PersonReID/DukeMTMC-ReID/pytorch/',
                                 '~/codes/OPP-PersonReID/MARS/pytorch/',
                                 '~/codes/OPP-PersonReID/MSMT17/pytorch/'],
                        help='path to the custom dataset')
    # dream parameters
    parser.add_argument('--r_feature', type=float, default=0.05,
                        help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--dr_lr', type=float, default=0.2, help='dreamer learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=10.0,
                        help='coefficient for the main loss in optimization')
    parser.add_argument('--ms', type=int, default=5000, help='memory size of dreamer')
    parser.add_argument('--iteration', type=int, default=1, help='optimization iteration')

    # other setting
    parser.add_argument('--temp', type=float, default=0.5, help='temperature for loss function')
    parser.add_argument('--T', default=2.0, type=float, help='temperature for target generation')
    parser.add_argument('--lamb', type=float, default=0.05, help='coefficient for the mix loss function')
    parser.add_argument('--sigma', type=float, default=1.0, help='parameter of Gaussian Kernel')
    parser.add_argument('--nearest', type=int, default=3, help='K nearest points')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--seed', default=0, type=int, help='for reproducibility')
    parser.add_argument('--msc', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()

    return opt
