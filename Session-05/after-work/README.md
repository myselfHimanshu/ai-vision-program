### Model 001

- Intuation
    - Model built after Model 003 in pervious directory
    - Keep architecture same as Model 003
    - Decrease number of channels
    - No transformation
    - add step-lr
    - add dropout

- Experiment Result
    - parameters : 7808
    - batch size : 128
    - lr : 0.01 with gamma = 0.1
    - dropout : 0.01 (in 2nd conv block)
    - training acc : 99.41
    - testing acc : 99.43 (in 11th epoch)
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/after-work/MNIST_model_final.ipynb)

- Analysis
    - parameters are under 8k
    - smoothest curve seen
    - 99.43 accuracy seen
    - Greater than 99.4 seen 6 times.
    - 99.42 accuracy seen in Epoch 8