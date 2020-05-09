import torch
from torchvision import datasets


class Cifar10DataLoader:
    def __init__(self, config, use_cuda):
        self.use_cuda = use_cuda
        self.config = config
        self.kwargs = {"num_workers":self.config.num_workers, 'pin_memory':True} if self.use_cuda else {}

    def __download_trainset(self, train_transforms):
        self.cifar10_trainset = datasets.CIFAR10(root="./data", train=True, download=True,
                                                    transform=train_transforms)
    
    def __download_testset(self, test_transforms):
        self.cifar10_testset = datasets.CIFAR10(root="./data", train=False, download=True,
                                                    transform=test_transforms)

    def get_train_loader(self, train_transforms):
        self.__download_trainset(train_transforms)
        train_loader = torch.utils.data.DataLoader(self.cifar10_trainset,
                                          batch_size=self.config.batch_size, shuffle=True, **self.kwargs)
    
        return train_loader

    def get_test_loader(self, test_transforms):
        self.__download_testset(test_transforms)
        test_loader = torch.utils.data.DataLoader(self.cifar10_testset,
                                          batch_size=self.config.batch_size, shuffle=True, **self.kwargs)
    
        return test_loader

    def get_classes(self):
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
        

