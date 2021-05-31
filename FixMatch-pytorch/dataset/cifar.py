import logging
import math
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class dataset_getters:
    def __init__(self, dataset, args, root):
        self.args = args
        self.root = root
        if dataset == 'cifar10':
            self.get_cifar10(args, root)
        elif dataset == 'cifar100':
            self.get_cifar100(args, root)
        else:
            print("wrong dataset name")
            exit(0)

    def get_cifar10_append(self, append_index_list):
        self.get_cifar10(self.args, self.root, append=True, append_index=append_index_list)

    def get_cifar10(self, args, root, append=False, append_index=None):
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        base_dataset = datasets.CIFAR10(root, train=True, download=True)
  
        if append is False:
            self.X_tr = base_dataset.data
            self.Y_tr = torch.from_numpy(np.array(base_dataset.targets))
            train_labeled_idxs, train_unlabeled_idxs = self.x_u_split(
                args, base_dataset.targets)
            self.train_labeled_idxs = train_labeled_idxs
            self.train_unlabeled_idxs = train_unlabeled_idxs
        else:
            self.train_labeled_idxs = np.concatenate((self.train_labeled_idxs, append_index))
            # self.train_labeled_idxs.extend(append_index)
            for i in range(50000):
                if i in self.train_labeled_idxs:
                    continue
                self.train_unlabeled_idxs = np.append(self.train_unlabeled_idxs, i)

        self.train_labeled_dataset = CIFAR10SSL(
            root, self.train_labeled_idxs, train=True,
            transform=transform_labeled)

        self.train_unlabeled_dataset = CIFAR10SSL(
            root, self.train_unlabeled_idxs, train=True,
            transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

        self.test_dataset = datasets.CIFAR10(
            root, train=False, transform=transform_val, download=False)

        # return train_labeled_dataset, train_unlabeled_dataset, test_dataset


    def get_cifar100(self, args, root, append=False):

        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

        base_dataset = datasets.CIFAR100(
            root, train=True, download=True)
        
        if append is False:
            train_labeled_idxs, train_unlabeled_idxs = x_u_split(
                args, base_dataset.targets)

        train_labeled_dataset = CIFAR100SSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)

        train_unlabeled_dataset = CIFAR100SSL(
            root, train_unlabeled_idxs, train=True,
            transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

        test_dataset = datasets.CIFAR100(
            root, train=False, transform=transform_val, download=False)

        return train_labeled_dataset, train_unlabeled_dataset, test_dataset


    def x_u_split(self, args, labels):
        label_per_class = args.num_labeled // args.num_classes
        labels = np.array(labels)
        labeled_idx = []
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        unlabeled_idx = np.array(range(len(labels)))
        for i in range(args.num_classes):
            idx = np.where(labels == i)[0]
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
        assert len(labeled_idx) == args.num_labeled

        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.eval_step / args.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]
    #     img = Image.fromarray(img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target
    def __getitem__(self, index_list):
        if isinstance(index_list, int):
            img, target = self.data[index_list], self.targets[index_list]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        img_list = []
        target_list = []
        for i in index_list:
            if self.transform is not None:
                img_list.append(self.transform(Image.fromarray(self.data[i])))

            if self.target_transform is not None:
                target_list.append(self.transform(self.targets[i]))
        return img_list, np.asarray(target_list), np.asarray(index_list)


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]
    #     img = Image.fromarray(img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target

    def __getitem__(self, index_list):
        if isinstance(index_list, int):
            img, target = self.data[index_list], self.targets[index_list]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        img_list = []
        target_list = []
        for i in index_list:
            if self.transform is not None:
                img_list.append(self.transform(Image.fromarray(self.data[i])))

            if self.target_transform is not None:
                target_list.append(self.transform(self.targets[i]))
        return img_list, np.asarray(target_list), np.asarray(index_list)

# DATASET_GETTERS = {'cifar10': get_cifar10,
#                    'cifar100': get_cifar100}
