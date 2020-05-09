from torchvision import transforms

class Cifar10Transforms(Object):
    def __init__(self):
        self.normalize = transforms.Normalize((.5,.5,.5),(.5,.5,.5))
        self.train_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def set_train_transforms(self, transforms):
        self.train_transform = transforms

    def set_test_transforms(self, transforms):
        self.test_transform = transforms

    def get_train_transforms(self):
        return self.train_transform
    
    def get_test_transforms(self):
        return self.test_transform