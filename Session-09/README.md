# Session 09 - DATA AUGMENTATION

## Objective

- Move transformations to Albumentations. 
    - Apply ToTensor, HorizontalFlip, Normalize + More.
- Test transforms are simple and only using ToTensor and Normalize
- Implement GradCam function. 
- Your final code must use imported functions to implement transformations and GradCam functionality
- Target Accuracy is 87%

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
    - Rotate,
    - RandomResizedCrop,
    - RandomBrightnessContrast
    - Normalize,
    - ToTensor

- Testing Albumentation Transformation (new test images was introduced for predicting single image purposes):
    - Resize,
    - Normalize,
    - ToTensor

- FILES
    - [Config FILE](https://github.com/myselfHimanshu/ultron-vision/blob/master/configs/cifar10_config.json)
    - [Albumenation transformation](https://github.com/myselfHimanshu/ultron-vision/blob/master/infdata/transformation/cifar10_tf.py)
    - [Resnet 18 model](https://github.com/myselfHimanshu/ultron-vision/blob/master/networks/resnet_net.py)
    - [Resnet agent](https://github.com/myselfHimanshu/ultron-vision/blob/master/agents/cifar10_agent.py)
    - [GradCam module](https://github.com/myselfHimanshu/ultron-vision/blob/master/utils/gradcam.py)
    - [LOGS FILE](https://github.com/myselfHimanshu/ultron-vision/blob/master/experiments/cifar10_exp_04_resnet_album/logs/exp_debug.log)

- Best Validataion Accuracy : 92.17%
- Total Epochs : 50

## Misclassified Images

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/misclassified_imgs.png)

## Accuracy Graph

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/accuracy.png)

## Loss Graph

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/loss.png)

## GRAD CAM RESULTS

For grad cam, I have introduced some images from outside cifar-10 dataset. Prediction are logged in log file and results for grad cam are shown below:

- CAR

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/grad_output_car.png)

- SHIP

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/grad_output_ship.png)

- TRUCK

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/grad_output_truck.png)

- DOG

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/grad_output_dog.png)

- BIRD

![](https://github.com/myselfHimanshu/ultron-vision/raw/master/experiments/cifar10_exp_04_resnet_album/stats/grad_output_bird.png)





    

