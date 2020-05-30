# Session 09 - quizDNN

## Objective

- Architecture

```
x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)
```

- Data : CIFAR10
- Target >= 75% in less than 40 Epochs


## Result

- FILES
    - [Config FILE](https://github.com/myselfHimanshu/ultron-vision/blob/quiz-dnn/configs/cifar10_quizdnn_config.json)
    - [QuizDNN network](https://github.com/myselfHimanshu/ultron-vision/blob/quiz-dnn/networks/quizz_net.py)
    - [CIFAR10 agent](https://github.com/myselfHimanshu/ultron-vision/blob/quiz-dnn/agents/cifar10_agent.py)
    - [LOGS FILE](https://github.com/myselfHimanshu/ultron-vision/blob/quiz-dnn/experiments/cifar10_exp_01_quizdnn/logs/exp_debug.log)

- Epochs used : 25
- Best Validation Accuracy : 85.61%


## Misclassified Images

![](https://github.com/myselfHimanshu/ultron-vision/raw/quiz-dnn/experiments/cifar10_exp_01_quizdnn/stats/misclassified_imgs.png)

## Accuracy Graph

![](https://github.com/myselfHimanshu/ultron-vision/raw/quiz-dnn/experiments/cifar10_exp_01_quizdnn/stats/accuracy.png)

## Loss Graph

![](https://github.com/myselfHimanshu/ultron-vision/raw/quiz-dnn/experiments/cifar10_exp_01_quizdnn/stats/loss.png)
