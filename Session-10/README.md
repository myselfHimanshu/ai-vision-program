# Session 10 - Advanced Concepts : Training and Learning Rates

## Objective

- Add CutOut augmentation. It should come from your transformations (albumentations)
- Implement LR Finder (for SGD, not for ADAM)
- Implement ReduceLROnPlatea
- Find best LR to train your model
- Use SDG with Momentum
- Train for 50 Epochs. 
- Show Training and Test Accuracy curves
- Target 88% Accuracy.
- Run GradCAM on the any 25 misclassified images. Make sure to mention what is the prediction and what was the ground truth label.

## MAIN CODE

The code has been shifted to new repo to keep things clean in this repo.

Link to repo : [ULTRON-VISION](https://github.com/myselfHimanshu/ultron-vision)

## RESULTS

- Validation Albumentaion Transformation:
    - Normalize,
    - ToTensor
    
- Train Albumentation Transformation:
    - HorizontalFlip,
    - Cutout,
    - RandomResizedCrop,
    - Normalize,
    - ToTensor

- FILES
    - [Config FILE](https://github.com/myselfHimanshu/ultron-vision/blob/master/experiments/cifar10_exp-06_resnet_album_findlr/summaries/config.txt)
    - [Albumenation transformation](https://github.com/myselfHimanshu/ultron-vision/blob/master/infdata/transformation/cifar10_tf.py)
    - [Resnet 18 model](https://github.com/myselfHimanshu/ultron-vision/blob/master/networks/resnet_net.py)
    - [CIFAR10 agent](https://github.com/myselfHimanshu/ultron-vision/blob/master/agents/cifar10_agent.py)
    - [CIFAR10 inference agent](https://github.com/myselfHimanshu/ultron-vision/blob/master/inference/cifar_iagent.py)
    - [LR finder module](https://github.com/myselfHimanshu/ultron-vision/blob/master/utils/lr_finder/lrfinder.py)
    - [LOGS FILE](https://github.com/myselfHimanshu/ultron-vision/blob/master/experiments/cifar10_exp-06_resnet_album_findlr/logs/exp_debug.log)

- Best Validataion Accuracy : 89.80%
- Total Epochs : 50

## Loss and Accuracy Graph

|Loss|Accuracy|
|--|--|
|<p align="center"><img width="80%" height="80%" src="https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp-06_resnet_album_findlr/stats/accuracy.png"/></p>|<p align="center"><img width="80%" height="80%" src="https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp-06_resnet_album_findlr/stats/loss.png"/></p>|

## Misclassified Images with Gradcam

True and predicted values are written on top of images (can be quite small font in here, just click it to open in new tab)

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp-06_resnet_album_findlr/stats/misclassified_imgs.png"/>
</p>


    

