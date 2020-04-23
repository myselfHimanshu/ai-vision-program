# MNIST DATASET PYTORCH MODEL FINAL SUBMISSION

## AIM :

- Highest Test Accuracy >= 99.4%
- Total Number of parameters <= 10,000
- Total Number of epochs <= 15

## Result :

### Model 001

- Intuation
    - Keep layers of model as simple as possible
    - No transformation
    - No regularization
    - No step-lr

- Experiment Result
    - parameters : 11930
    - batch size : 128
    - lr : 0.01
    - dropout : none
    - batch-norm : none
    - epoch : 15
    - training acc : 98.85
    - testing acc : 98.50
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/notebooks/MNIST_model_001.ipynb)

- Analysis
    - huge gap between training and testing loss
    - testing line plot indicates high learning rate used
    - training accuracy is low, more scope for learning

### Model 002

- Intuation
    - Keep architecture same as above
    - Try to overfit the model
    - No transformation
    - Add batch-norm to every layer
    - No step-lr
    - decrease lr

- Experiment Result
    - parameters : 12122
    - batch size : 128
    - lr : 0.001
    - training acc : 99.57
    - testing acc : 99.31
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/notebooks/MNIST_model_002.ipynb)

- Analysis
    - model is large, but working
    - we see overfitting, that's a good sign. Now we can work on regularizations
    - the learning rate seems to be working
    
### Model 003

- Intuation
    - Keep layers of model as simple as possible
    - No transformation
    - No step-lr
    - decrease number of parameters

- Experiment Result
    - parameters : 8138
    - batch size : 128
    - lr : 0.001
    - training acc : 99.51
    - testing acc : 99.35
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/notebooks/MNIST_model_003.ipynb)

- Analysis
    - parameters are under 10k
    - we see overfitting,
    - need to increase the regularization strength
    - accuracy line plots are not smooth

### Model 004

- Intuation
    - Keep architecture same as above
    - No transformation
    - No step-lr
    - increase model capacity
    - add dropout

- Experiment Result
    - parameters : 8766
    - batch size : 128
    - lr : 0.001
    - dropout : 0.05 (in 2nd conv block)
    - training acc : 99.35
    - testing acc : 99.32
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/notebooks/MNIST_model_004.ipynb)

- Analysis
    - parameters are under 10k
    - model didn't train well like the previous one
    - loss graph is slightly smooth, need to increase the regularization strength
    - haven't seen 99.4 on test dataset

### Model 005

- Intuation
    - Keep architecture same as above
    - add a transformation
    - No step-lr
    - increase model capacity
    - work around lr

- Experiment Result
    - parameters : 9148
    - batch size : 128
    - lr : 0.01
    - momentum : 0.9
    - dropout : 0.05 (in 2nd conv block)
    - training acc : 99.48
    - testing acc : 99.45
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/notebooks/MNIST_model_005.ipynb)

- Analysis
    - parameters are under 10k
    - smoothest curve seen
    - 99.45 accuracy seen
    - no underfitting or overfitting
    


