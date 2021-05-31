import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from copy import deepcopy
import pdb
from torchvision import transforms
from PIL import Image
# import resnet
# from dataset import get_handler

def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    else:
        return DataHandler4
        
class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class Strategy:
    def __init__(self, X, Y, idxs_lb, model):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.model = model
        self.handler = get_handler('CIFAR10')
        # self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # def _train(self, epoch, loader_tr, optimizer):
    #     self.clf.train()
    #     accFinal = 0.
    #     for batch_idx, (x, y, idxs) in enumerate(loader_tr):
    #         x, y = Variable(x.cuda()), Variable(y.cuda())
    #         optimizer.zero_grad()
    #         out, e1 = self.clf(x)
    #         loss = F.cross_entropy(out, y)
    #         accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
    #         loss.backward()

    #         # clamp gradients, just in case
    #         for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

    #         optimizer.step()
    #     return accFinal / len(loader_tr.dataset.X)


    
    # def train(self):
    #     def weight_reset(m):
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             m.reset_parameters()

    #     n_epoch = self.args['n_epoch']
    #     self.clf =  self.net.apply(weight_reset).cuda()
    #     optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

    #     idxs_train = np.arange(self.n_pool)[self.idxs_lb]
    #     loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
   
    #     epoch = 1
    #     accCurrent = 0.
    #     while accCurrent < 0.99: 
    #         accCurrent = self._train(epoch, loader_tr, optimizer)
    #         epoch += 1
    #         print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True, end='\r')
    #         if (epoch % 50 == 0) and (accCurrent < 0.2): # reset if not converging
    #             self.clf = self.net.apply(weight_reset).cuda()
    #             optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

    #     print("")

    # def predict(self, X, Y):
    #     if type(X) is np.ndarray:
    #         loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
    #                         shuffle=False, **self.args['loader_te_args'])
    #     else: 
    #         loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
    #                         shuffle=False, **self.args['loader_te_args'])

    #     self.clf.eval()
    #     P = torch.zeros(len(Y)).long()
    #     with torch.no_grad():
    #         for x, y, idxs in loader_te:
    #             x, y = Variable(x.cuda()), Variable(y.cuda())
    #             out, e1 = self.clf(x)
    #             pred = out.max(1)[1]
    #             P[idxs] = pred.data.cpu()
    #     return P

    # def predict_prob(self, X, Y):
    #     loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
    #     self.clf.eval()
    #     probs = torch.zeros([len(Y), len(np.unique(self.Y))])
    #     with torch.no_grad():
    #         for x, y, idxs in loader_te:
    #             x, y = Variable(x.cuda()), Variable(y.cuda())
    #             out, e1 = self.clf(x)
    #             prob = F.softmax(out, dim=1)
    #             probs[idxs] = prob.cpu().data
        
    #     return probs

    # def predict_prob_dropout(self, X, Y, n_drop):
    #     loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
    #                         shuffle=False, **self.args['loader_te_args'])

    #     self.clf.train()
    #     probs = torch.zeros([len(Y), len(np.unique(Y))])
    #     with torch.no_grad():
    #         for i in range(n_drop):
    #             print('n_drop {}/{}'.format(i+1, n_drop))
    #             for x, y, idxs in loader_te:
    #                 x, y = Variable(x.cuda()), Variable(y.cuda())
    #                 out, e1 = self.clf(x)
    #                 prob = F.softmax(out, dim=1)
    #                 probs[idxs] += prob.cpu().data
    #     probs /= n_drop
        
    #     return probs

    # def predict_prob_dropout_split(self, X, Y, n_drop):
    #     loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
    #                         shuffle=False, **self.args['loader_te_args'])

    #     self.clf.train()
    #     probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
    #     with torch.no_grad():
    #         for i in range(n_drop):
    #             print('n_drop {}/{}'.format(i+1, n_drop))
    #             for x, y, idxs in loader_te:
    #                 x, y = Variable(x.cuda()), Variable(y.cuda())
    #                 out, e1 = self.clf(x)
    #                 probs[i][idxs] += F.softmax(out, dim=1).cpu().data
    #         return probs

    # def get_embedding(self, X, Y):
    #     loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
    #                         shuffle=False, **self.args['loader_te_args'])
    #     self.clf.eval()
    #     embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
    #     with torch.no_grad():
    #         for x, y, idxs in loader_te:
    #             x, y = Variable(x.cuda()), Variable(y.cuda())
    #             out, e1 = self.clf(x)
    #             embedding[idxs] = e1.data.cpu()
        
    #     return embedding

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.model
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        loader_te = DataLoader(self.handler(X, Y, transform=transform_labeled),
                            shuffle=False)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout = model(x)
                out = model.get_emb(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)

