# CIFAR10_TENSORFLOW_CNN

## github
https://github.com/benlin131020/CIFAR10_TENSORFLOW_CNN

## 網路架構
tensorboard_network_structure.png

## accuracy & loss
accuracy.PNG

loss.PNG

training data 上的 accuracy 及 loss 尚有上升及下降的空間，需要更多的 epoch 訓練。

testing data 則有 overfitting 的現象，需要更多 dropout 及 batch normalization。

## tf.data.Dataset
因為 Keras 提供的 cifar10 沒有 batch 的功能，因此利用 `tf.data.Dataset` 提供的 batch 功能。

詳見程式碼內的 comment 及 https://www.tensorflow.org/guide/datasets

## 開啟tensorboard
```
cd hw3
tensorboard --logdir logs
```
