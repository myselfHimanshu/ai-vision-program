from __future__ import print_function
import json
import os
import pandas as pd

curr_dir = os.path.dirname(__file__)

from utils import *
from networks.resnet import ResNet18
from models.cifar10agent import Cifar10Agent
from data.transformation.cifar10transforms import Cifar10Transforms
from dataset.cifar10dataloader import Cifar10DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchsummary import summary
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class Main:
    def __init__(self, config_path):
        """
        Initialize the network
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.device, self.use_cuda = check_cuda()

        # define dataset loader
        self.loader = Cifar10DataLoader(self.config, self.use_cuda)

        # get train and test loader
        transf = Cifar10Transforms()
        train_transforms = transf.get_train_transforms()
        test_transforms = transf.get_test_transforms()

        self.train_loader = self.loader.get_train_loader(train_transforms)
        self.test_loader = self.loader.get_test_loader(test_transforms)

        # get class names
        self.classes = self.loader.get_classes()

        # load network
        self.net = ResNet18().to(self.device)
        summary(self.net, input_size=tuple(self.config['input_size']))

        #criterion
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

        # stepLR
        if(self.config["scheduler"]):
            self.schedular = StepLR(self.optimizer, step_size=self.config["step_size"], gamma=self.config["gamma"])

        self.MODEL_PATH = os.path.join(curr_dir, self.config["checkpoint_dir"],"best_weights.pt")
        self.model = Cifar10Agent(self.net, self.device, self.criterion, self.optimizer, self.config["l1_decay"], self.config["l2_decay"])


class TrainNetwork(Main):
    def __init__(self, config_path):
        super().__init__(config_path)

    def start(self):
        """
        Start training the network
        """
        for epoch in range(1, self.config["epochs"]+1):
            print(f"\nEPOCH : {epoch}\n")
            self.model.train(self.train_loader)
            if self.config["scheduler"]:
                self.schedular.step()
            self.model.test(self.test_loader, epoch, self.MODEL_PATH)

        result = {f'{self.config["model_name"]}':{'train_losses':self.model.train_losses, 'test_losses':self.model.test_losses,
                            'train_acc':self.model.train_acc, 'test_acc':self.model.test_acc}}

        with open(self.config["acc_loss_data_file"], "w") as f:
            json.dump(result, f)

    def visulaize_dataset(self):
        """
        visualize dataset before training
        """
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()

        visulaize_data(images, labels, self.classes)


class PostAnalysis(Main):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.net, self.optimizer, epoch, val_max_acc, misclassified_images = load_ckp(self.MODEL_PATH, self.net, self.optimizer)
        self.y_pred_list = []
        self.y_list = []
        self.id2classes = {i:y for i,y in enumerate(self.classes)}
    
    def show_validation_graph(self, type_='acc'):
        epoch_count = range(1, self.config['epochs']+1)
        plot_graphs(self.config['model_name'], epoch_count, self.config['acc_loss_data_file'], type_)

    def show_misclassified_images(self):
        images = get_misclassified_images(self.net, self.optimizer, self.MODEL_PATH)
        plot_misclassified_images(self.config['model_name'], images, self.id2classes)

    def show_per_class_accuracy(self):

        confusion_matrix = torch.zeros(self.config['num_classes'], self.config['num_classes'])
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                classes = classes.to(self.device)
                
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
        class_acc = (100.*confusion_matrix.diag()/confusion_matrix.sum(1)).cpu().numpy()
        class_acc = zip(self.classes, class_acc)
        for class_name, acc_score in class_acc:
          print(f"{class_name}\t\t{acc_score:4f}")


if __name__=="__main__":
    config_file = os.path.join(curr_dir, "configs/cifar10_config.json")
    trainNetwork = TrainNetwork(config_file)

    trainNetwork.visulaize_dataset()
    trainNetwork.start()

    analysis = PostAnalysis(config_file)
    analysis.show_validation_graph()
    analysis.show_misclassified_images()
    analysis.show_per_class_accuracy()