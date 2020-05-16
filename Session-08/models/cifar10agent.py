import sys
import os

curr_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(curr_dir, "../"))

import torch
import torch.nn.functional as F

from utils import *

from tqdm import tqdm
import numpy as np


class Cifar10Agent:
    def __init__(self, net, device, criterion, optimizer, l1_decay=0.0, l2_decay=0.0):
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        self.misclassified_images = {}

        self.l1_decay = l1_decay
        self.l2_decay = l2_decay

        self.minimum_test_loss = np.Inf
        self.maximum_test_acc = 0.0


    def train(self, train_loader):
        running_loss = 0.0
        running_correct = 0

        self.net.train()

        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)

            if self.l1_decay>0.0:
                loss += regularize_loss(self.net, loss, self.l1_decay, 1)
            if self.l2_decay>0.0:
                loss += regularize_loss(self.net, loss, self.l2_decay, 2)

            _, preds = torch.max(output.data, 1)
            loss.backward()
            self.optimizer.step()

            #calculate training running loss
            running_loss += loss.item()
            running_correct += (preds == target).sum().item()
            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

        r_total_loss = running_loss/len(train_loader.dataset)
        r_total_acc = 100. * running_correct/len(train_loader.dataset)

        self.train_losses.append(r_total_loss)
        self.train_acc.append(r_total_acc)
        print("\n")
        print(f"  TRAIN avg loss: {r_total_loss:.4f} train acc: {r_total_acc:.4f}\n")

    def test(self, test_loader, epoch, checkpoint_fpath = None):
        running_loss = 0.0
        running_correct = 0

        self.net.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                running_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                
                is_correct = pred.eq(target.view_as(pred))
                misclass_indx = (is_correct==0).nonzero()[:,0]
                for indx in misclass_indx:
                    if str(epoch) not in self.misclassified_images:
                        self.misclassified_images[str(epoch)] = []
                    self.misclassified_images[str(epoch)].append({
                        "target" : target[indx],
                        "pred" : pred[indx],
                        "img" : data[indx]
                    })

                running_correct += pred.eq(target.view_as(pred)).sum().item()

        r_total_loss = running_loss/len(test_loader.dataset)
        r_total_acc = 100.*running_correct/len(test_loader.dataset)

        if(r_total_acc>=self.maximum_test_acc):
            self.maximum_test_acc = r_total_acc
            if checkpoint_fpath:
                self.save_checkpoint(epoch, checkpoint_fpath)
                print(f"  Best Model Saved!!!\n")
            else:
                print(f"  Couldn't save the model. Path not defined!!!\n")


        self.test_losses.append(r_total_loss)
        self.test_acc.append(r_total_acc)

        print("\n")
        print(f"  TEST avg loss: {r_total_loss:.4f} test acc: {r_total_acc:.4f}\n")

    def save_checkpoint(self, epoch, checkpoint_fpath):
        checkpoint = {
                'epoch' : epoch,
                'misclassified_images' : self.misclassified_images[str(epoch)],
                'valid_max_acc': self.maximum_test_acc,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        save_ckp(checkpoint, checkpoint_fpath)