import torch
import torchvision

import os
curr_dir = os.path.dirname(__file__)

import numpy as np
import matplotlib.pyplot as plt
import json

def check_cuda():
    """
    check for cuda
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
        print(f"Number of GPU's available : {torch.cuda.device_count()}")
        print(f"GPU device name : {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU instead")
        device = torch.device("cpu")
        use_cuda = False
    
    return device, use_cuda

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def visulaize_data(images, target=None, classes=None, n=30):
    """
    Visualize dataset
    """
    figure = plt.figure(figsize=(10,10))

    for i in range(1, n+1):
        plt.subplot(5,n//5,i)
        plt.axis('off')
        imshow(images[i-1])
        plt.title("Actual : {}".format(classes[target[i-1]]))

    plt.tight_layout()

def regularize_loss(model, loss, decay, norm_value):
    """
    L1/L2 Regularization
    decay : l1/l2 decay value
    norm_value : the order of norm
    """
    r_loss = 0
    # get sum of norm of parameters
    for param in model.parameters():
        r_loss += torch.norm(param, norm_value)
    # update loss value
    loss += decay * r_loss

    return loss

def save_ckp(state, checkpoint_fpath):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save model
    """
    f_path = checkpoint_fpath
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # get epoch
    epoch = checkpoint['epoch']
    # get val_max_acc
    val_max_acc = checkpoint['valid_max_acc']
    # get misclassified images
    misclassified_images = checkpoint['misclassified_images']
    # return model, optimizer, epoch, val_max_acc, misclassified_images
    return model, optimizer, epoch, val_max_acc, misclassified_images

def get_misclassified_images(net, optimizer, ckp_path):
    """
    load best model and return misclassified images
    """
    _, misclassified_images = load_ckp(ckp_path, net, optimizer)
    return misclassified_images

def validation_stat(model_name, file_path, type_='acc'):
    """
    load and return accuracy and losses numbers per epoch
    """
    with open(file_path) as f:
        data = json.load(f)

    if type_=="acc":
        return data[f"{model_name}"]["test_acc"] 
    else:
        return data[f"{model_name}"]["test_losses"]

def plot_graphs(model_name, epoch_count, file_path, type_='acc'):
    """
    plot accuracy or losses graphs
    """
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(epoch_count, validation_stat(model_name, file_path, type_))

    plt.xlabel('Epoch')
    plt.ylabel(type_)
    plt.show();

    fig.savefig(os.path.join(curr_dir, f'images/validation_%s.png' % (type_)))

def plot_misclassified_images(model_name, images, id2class, n=25):
    """
    plot misclassified images from best model saved
    """
    figure = plt.figure(figsize=(10,10))

    for i in range(1, n+1):
        plt.subplot(5,n//5,i)
        plt.axis('off')
        imshow(images[i-1]["img"])
        plt.title("Predicted : {} \nActual : {}".format(id2class[images[i-1]["pred"][0].cpu().numpy()[0]], id2class[images[i-1]["target"].cpu().numpy()[0]]))

    plt.tight_layout()
    plt.savefig(os.path.join(curr_dir, f"images/{model_name}_image.png"))

    

