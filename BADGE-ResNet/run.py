import numpy as np
import sys
import gzip
# import openml
import os
import argparse
from dataset import get_dataset, get_handler
from model import get_net
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import pdb
from scipy.stats import zscore

from query_strategies import RandomSampling, BadgeSampling, \
                                BaselineSampling, LeastConfidence, MarginSampling, \
                                EntropySampling, CoreSet, ActiveLearningByLearning, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='badge')
parser.add_argument('--name', help='test_name', type=str, default='test')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='resnet')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='CIFAR10')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=1000)
parser.add_argument('--nStart', help='number of points to start', type=int, default=1000)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
opts = parser.parse_args()

# parameters
# NUM_INIT_LB = opts.nStart
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data

# data defaults
args_pool = {'MNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'FashionMNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10':
                {'n_epoch': 3, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}
                }
args_pool['CIFAR10'] = {'n_epoch': 3, 
    'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,     0.2435, 0.2616))]),
    'loader_tr_args':{'batch_size': 128, 'num_workers': 3},
    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
    'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
    'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])    
}

opts.nClasses = 10
args_pool['CIFAR10']['transform'] =  args_pool['CIFAR10']['transformTest'] # remove data augmentation
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']

if opts.did == 0: args = args_pool[DATA_NAME]
if not os.path.exists(opts.path):
    os.makedirs(opts.path)

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
opts.dim = np.shape(X_tr)[1:]
handler = get_handler(opts.data)

args['lr'] = opts.lr

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)

print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# # linear model class
# class linMod(nn.Module):
#     def __init__(self, nc=1, sz=28):
#         super(linMod, self).__init__()
#         self.lm = nn.Linear(int(np.prod(dim)), opts.nClasses)
#     def forward(self, x):
#         x = x.view(-1, int(np.prod(dim)))
#         out = self.lm(x)
#         return out, x
#     def get_embedding_dim(self):
#         return int(np.prod(dim))

# mlp model class
# class mlpMod(nn.Module):
#     def __init__(self, dim, embSize=256):
#         super(mlpMod, self).__init__()
#         self.embSize = embSize
#         self.dim = int(np.prod(dim))
#         self.lm1 = nn.Linear(self.dim, embSize)
#         self.lm2 = nn.Linear(embSize, opts.nClasses)
#     def forward(self, x):
#         x = x.view(-1, self.dim)
#         emb = F.relu(self.lm1(x))
#         out = self.lm2(emb)
#         return out, emb
#     def get_embedding_dim(self):
#         return self.embSize

# load specified network
# if opts.model == 'mlp':
#     net = mlpMod(opts.dim, embSize=opts.nEmb)

if opts.model == 'resnet':
    net = resnet.ResNet18()
    # # TODO : change to FixMatch ResNext
    # if opts.data == 'CIFAR10':
    #     num_classes = 10
    #     model_cardinality = 8
    #     model_depth = 29
    #     model_width = 64
    # elif opts.data == "CIFAR100":
    #     num_classes = 100
    #     model_cardinality = 4
    #     model_depth = 28
    #     model_width = 4

    # import resnext as model
    # net = model.build_resnext(model_cardinality, model_depth, model_width, num_classes)
elif opts.model == 'wideresnet':
    import wideresnet 
    if opts.data == 'CIFAR10':
        model_depth = 28
        model_width = 2
        num_classes = 10
    else:
        model_depth = 28
        model_width = 8
        num_classes = 100
    net = wideresnet.build_wideresnet(depth=model_depth,
                                            widen_factor=model_width,
                                            dropout=0,
                                            num_classes=num_classes)
# elif opts.model == 'vgg':
#     net = vgg.VGG('VGG16')
else: 
    print('choose a valid model - mlp, resnet, or vgg', flush=True)
    raise ValueError

# if opts.did > 0 and opts.model != 'mlp':
#     print('openML datasets only work with mlp', flush=True)
#     raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

# set up the specified sampler
if opts.alg == 'rand': # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf': # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'marg': # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge': # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset': # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'entropy': # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'baseline': # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'albl': # active learning by learning
    albl_list = [LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args),
        CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
else: 
    print('choose a valid acquisition function', flush=True)
    raise ValueError

# print info
print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

# testing accuracy log
acc_log = []
# round 0 accuracy
strategy.train()
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print(str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)
acc_log.append(acc[0])

# NUM_ROUND = 2
for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd), flush=True)

    # query
    output = strategy.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # report weighted accuracy
    corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), flush=True)
    acc_log.append(acc[rd])

    if sum(~strategy.idxs_lb) < opts.nQuery: 
        # save log
        if not os.path.exists("./output/"):
            os.mkdir("./output/")

        directory = "./output/" + opts.alg + "/"
        if not os.path.exists(directory):
            os.mkdir(directory)

        np.save(directory+"testAcc.npy", np.array(acc_log))
        print(acc_log)
        sys.exit('too few remaining points to query')

print("for loop finished")
# save log
if not os.path.exists("./output/"):
	os.mkdir("./output/")

directory = "./output/" + opts.name + "/"
if not os.path.exists(directory):
	os.mkdir(directory)

np.save(directory+"testAcc.npy", np.array(acc_log))
print(acc_log)




