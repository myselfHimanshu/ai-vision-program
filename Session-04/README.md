# MNIST DATASET PYTORCH MODEL

AIM :

- Test Accuracy >= 99.4%
- Total Number of parameters <= 20,000
- Total Number of epochs <= 20

## EXPERIMENTS

### MODEL 02

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             144
         Dropout2d-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,304
         Dropout2d-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           2,304
         Dropout2d-9           [-1, 16, 14, 14]               0
      BatchNorm2d-10           [-1, 16, 14, 14]              32
           Conv2d-11           [-1, 16, 14, 14]           2,304
        Dropout2d-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
        MaxPool2d-14             [-1, 16, 7, 7]               0
           Conv2d-15             [-1, 32, 7, 7]           4,608
        Dropout2d-16             [-1, 32, 7, 7]               0
      BatchNorm2d-17             [-1, 32, 7, 7]              64
           Conv2d-18             [-1, 16, 7, 7]           4,608
        Dropout2d-19             [-1, 16, 7, 7]               0
      BatchNorm2d-20             [-1, 16, 7, 7]              32
           Conv2d-21             [-1, 10, 5, 5]           1,450
        AvgPool2d-22             [-1, 10, 1, 1]               0
================================================================

Total params: 17,946
Trainable params: 17,946
Non-trainable params: 0

Epoch < 20

Test Accuracy : 99.440%

### MODEL 03

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
         Dropout2d-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,320
         Dropout2d-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           2,320
         Dropout2d-9           [-1, 16, 14, 14]               0
      BatchNorm2d-10           [-1, 16, 14, 14]              32
           Conv2d-11           [-1, 16, 14, 14]           2,320
        Dropout2d-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
        MaxPool2d-14             [-1, 16, 7, 7]               0
           Conv2d-15             [-1, 32, 7, 7]           4,640
        Dropout2d-16             [-1, 32, 7, 7]               0
      BatchNorm2d-17             [-1, 32, 7, 7]              64
           Conv2d-18             [-1, 16, 7, 7]           4,624
        Dropout2d-19             [-1, 16, 7, 7]               0
      BatchNorm2d-20             [-1, 16, 7, 7]              32
           Conv2d-21             [-1, 10, 5, 5]           1,450
        AvgPool2d-22             [-1, 10, 1, 1]               0
================================================================

Total params: 18,058
Trainable params: 18,058
Non-trainable params: 0

Epoch < 20

Test Accuracy : 99.460%

### MODEL 04

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
         Dropout2d-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,320
         Dropout2d-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           2,320
         Dropout2d-9           [-1, 16, 14, 14]               0
      BatchNorm2d-10           [-1, 16, 14, 14]              32
           Conv2d-11           [-1, 16, 14, 14]           2,320
        Dropout2d-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
        MaxPool2d-14             [-1, 16, 7, 7]               0
           Conv2d-15             [-1, 16, 7, 7]           2,320
        Dropout2d-16             [-1, 16, 7, 7]               0
      BatchNorm2d-17             [-1, 16, 7, 7]              32
           Conv2d-18             [-1, 16, 7, 7]           2,320
        Dropout2d-19             [-1, 16, 7, 7]               0
      BatchNorm2d-20             [-1, 16, 7, 7]              32
           Conv2d-21             [-1, 10, 5, 5]           1,450
        AvgPool2d-22             [-1, 10, 1, 1]               0
================================================================

Total params: 13,402
Trainable params: 13,402
Non-trainable params: 0

Epoch < 20

Test Accuracy : 99.460%

### MODEL 05

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             144
         Dropout2d-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,304
         Dropout2d-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           2,304
         Dropout2d-9           [-1, 16, 14, 14]               0
      BatchNorm2d-10           [-1, 16, 14, 14]              32
           Conv2d-11           [-1, 16, 14, 14]           2,304
        Dropout2d-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
        MaxPool2d-14             [-1, 16, 7, 7]               0
           Conv2d-15             [-1, 16, 5, 5]           2,304
        Dropout2d-16             [-1, 16, 5, 5]               0
      BatchNorm2d-17             [-1, 16, 5, 5]              32
           Conv2d-18             [-1, 32, 3, 3]           4,608
        Dropout2d-19             [-1, 32, 3, 3]               0
      BatchNorm2d-20             [-1, 32, 3, 3]              64
        AvgPool2d-21             [-1, 32, 1, 1]               0
           Linear-22                   [-1, 10]             330
================================================================

Total params: 14,522
Trainable params: 14,522
Non-trainable params: 0

Epoch < 20

Test Accuracy : 99.440%

