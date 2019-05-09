# CIFAR10_TENSORFLOW_CNN

## github
https://github.com/benlin131020/CIFAR10_TENSORFLOW_CNN

## 網路架構
tensorboard_network_structure.png

## accuracy & loss
accuracy.PNG

loss.PNG

training data 上的 accuracy 及 loss 尚有上升及下降的空間，需要更多的epoch訓練。

testing data 則有 overfitting 的現象，需要更多dropout 及 batch normalization。

## 開啟tensorboard
cd hw3
tensorboard --logdir logs
