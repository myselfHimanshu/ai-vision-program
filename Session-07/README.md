# Advanced Convolutions

## CIFAR10 Dataset

### Objective

- Achieve an accuracy of greater than 85% on CIFAR-10 dataset
    - architecture ResNet18

### Template Structure

```
.
├── checkpoints // store trained models
├── configs // store networks configuration parameters
│ ├── cifar10_config.json
│ └── resnet_config.json
├── data // define our dataset
│ └── transformation // custom transformation, e.g. data augmentation
├── dataset // the data loader
│ └── cifar10dataloader.py
├── images // save images
├── losses // custom losses
├── main.py
├── models // define training and testing
│ ├── base.py
│ └── cifar10agent.py
├── networks // define our network
│ ├── cifar10net.py
│ ├── resnet.py
│ └── utils.py
├── notebooks // jupyter notebooks
│ └── 001_main_89_78.ipynb
├── README.md
├── requirements.txt
└── utils.py
```

### Model Summary

- Total params: 11,173,962
- Forward/backward pass size (MB): 11.25
- Params size (MB): 42.63
- Estimated Total Size (MB): 53.89

### Experiment Result

- parameters : 11,173,962
- batch size : 128
- lr : 0.1 (step lr, gamma=0.1, step_size=15)
- epoch : 50
- training acc : 98.65%
- testing acc : 89.78%
- [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-08/notebooks/001_main_89_78.ipynb)

### Validation Plots

#### Losses

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-08/images/validation_loss.png"/>
</p>

#### Accuracy

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-08/images/validation_acc.png"/>
</p>

### Misclassified Images

<p align="center">
  <img src="https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-08/images/misclassified_images.png"/>
</p>

### TODO

- [ ] requirements.txt
- [ ] custom_loss_function
