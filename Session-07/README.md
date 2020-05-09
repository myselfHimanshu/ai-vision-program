# Advanced Convolutions

## CIFAR10 Dataset

### Objective

- Achieve an accuracy of greater than 80% on CIFAR-10 dataset
    - architecture to C1C2C3C40 (basically 3 MPs)
    - total params to be less than 1M
    - RF must be more than 44
    - one of the layers must use Depthwise Separable Convolution
    - one of the layers must use Dilated Convolution
    - use GAP

### Template Structure

```
.
├── checkpoints // store trained models
├── configs // store networks configuration parameters
│ └── cifar10_config.json
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
│ └── utils.py
├── notebooks // jupyter notebooks
│ └── cifar10.ipynb
├── README.md
├── requirements.txt
└── utils.py
```

### Model Summary

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-07/images/network.png"/>
</p>

### Notebook

- [link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-07/notebooks/002_main_85_55.ipynb)

### Validation Plots

#### Losses

![network](https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-07/images/validation_loss.png)

#### Accuracy

![network](https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-07/images/validation_acc.png)

### Misclassified Images

![network](https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-07/images/misclassified_images.png)
