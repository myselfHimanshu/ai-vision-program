# Session 11 - SUPER-CONVERGENCE

## Objective

- Write a code that draws zig-zag curve with upper and lower limit.
- Write a shallow 3-layer architecture
    - PrepLayer = Conv 3x3 (s1, p1) >> BN >> RELU, 64k
    - Layer1
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU, 128k
        - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU) )(X), 128k
        - Add(X, R1)
    - Layer2
        - Conv 3x3, >> MaxPool2D >> BN >> RELU, 256k
    - Layer3
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU, 512k
        - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU) )(X), 512k
        - Add(X, R2)
    - MaxPooling with Kernel Size 4
    - FC Layer >> Softmax
- Use One Cycle Policy
    - Total epochs = 24
    - Max at epoch = 5
    - LRMIN = ?
    - LRMAX = ?
    - No Annihilation
- Augmentation
    - RandomCrop 32, 32 (after padding of 4)
    - FlipLR
    - CutOut(8, 8)
- Batch_Size = 512
- Target 90% Accuracy.

## MAIN CODE

The code has been shifted to new repo to keep things clean in this repo.

Link to repo : [ULTRON-VISION](https://github.com/myselfHimanshu/ultron-vision/tree/session-11)

## RESULTS

- Max-LR was found using LR-Range test
    - Max learning rate : 0.007
- For one_cycle policy, div_factor = 10
    - initial learning rate : 0.007/10 = 0.0007
- As no annihilation was required, final_div_factor = 1
    - min learning rate : 0.0007

The configuration for the experiment can be found in `config file`. LR range test is performed and code can be found in `CIFAR10 agent` file. The training and validation loss and accuracy logs can be found in `LOGS` file.

- FILES
    - [Config FILE](https://github.com/myselfHimanshu/ultron-vision/blob/session-11/experiments/cifar10_session11-exp-002/summaries/config.txt)
    - [Albumenation transformation](https://github.com/myselfHimanshu/ultron-vision/blob/session-11/infdata/transformation/cifar10_tf.py)
    - [3Layer-Network](https://github.com/myselfHimanshu/ultron-vision/blob/session-11/networks/threelayer_net.py)
    - [CIFAR10 agent](https://github.com/myselfHimanshu/ultron-vision/blob/session-11/agents/cifar10_agent.py)
    - [CIFAR10 inference agent](https://github.com/myselfHimanshu/ultron-vision/blob/session-11/inference/cifar_iagent.py)
    - [LOGS FILE](https://github.com/myselfHimanshu/ultron-vision/blob/session-11/experiments/cifar10_session11-exp-002/logs/exp_debug.log)

- Best Validataion Accuracy : 91.02%
- Total Epochs : 24

## Loss and Accuracy Graph

|Loss|Accuracy|
|--|--|
|<p align="center"><img width="80%" height="80%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-11/experiments/cifar10_session11-exp-002/stats/accuracy.png"/></p>|<p align="center"><img width="80%" height="80%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-11/experiments/cifar10_session11-exp-002/stats/loss.png"/></p>|

## Misclassified Images with Gradcam

True and predicted values are written on top of images (can be quite small font in here, just click it to open in new tab)

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/ultron-vision/raw/session-11/experiments/cifar10_session11-exp-002/stats/misclassified_imgs.png"/>
</p>

## ZIGZAG CURVE

- Link to file : [ZIG-ZAG-Curve]()

![](https://github.com/myselfHimanshu/ai-vision-program/raw/master/Session-11/zigzag_curve/zigzag_curve.png)
    

