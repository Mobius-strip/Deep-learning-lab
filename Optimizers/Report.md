# Introduction

## Problem Statement
The goal is to use ResNet18 pre-trained on ImageNet (Pre-trained models found online) and finetune
the model on the CIFAR100 dataset for classification task and plot curves for training loss and training
accuracy and report the final top-5 test accuracy. The three optimizers used are listed in the following
list:
1. Adam
2. Adagrad
3. RMSprop
The final result is expected to be useful for analyzing which optimizer performs the best. The
hyperparameters of each optimizer are varied.

## Metrics
Top-5 accuracy means any of our model’s top 5 highest probability answers match with the
expected answer. it considers a classification correct if any of the five predictions match the target
label
Top5 acc =

N1i=1∑N[1≤j≤5maxP(i,j), where C(i) is in the top-
5 predictions for sample i

## Analysis

### Data

The CIFAR-100 dataset consists of 60000 32x32 color images in 100 classes, with 600 images per class.
The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine"
label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs). There are
50000 training images and 10000 test images. The metafile contains the label names of each class and
superclass.

## Methodology

## Data Transformation
The preprocessing done in the transformation of images after downloading consists of the following
steps:
1. Random Horizontal Flip
2. Random Cropping
3. Conversion to tensor
4. Normalization

## Implementation
The implementation process can be split into two main stages:
1. Loading the pretrained Resnet18 model
2. Varying the parameters, optimizers and classifying

# Results 


<img width="384" alt="image" src="https://github.com/Mobius-strip/Deep-learning-lab/assets/91667232/28fa450c-92db-4706-8075-360834cf876f">
<img width="369" alt="image" src="https://github.com/Mobius-strip/Deep-learning-lab/assets/91667232/5057891a-6f25-41d5-a1cd-8889a3f08026">
<img width="368" alt="image" src="https://github.com/Mobius-strip/Deep-learning-lab/assets/91667232/363b760f-2434-491c-86ea-bb36dc546280">

| RMSprop | Index | Learning rate | Beta | Weight_decay | Top-5 accuracy |
|---------|-------|---------------|------|--------------|----------------|
|         | 1     | 0.1           | 0.9  | 0.1          | 57.84%         |
|         | 2     | 0.1           | 0.9  | 0.001        | 68.03%         |
|         | 3     | 0.1           | 0.9  | 0.01         | 55.06%         |
|         | 4     | 0.1           | 0.95 | 0.1          | 58.73%         |
|         | 5     | 0.1           | 0.95 | 0.001        | 62.89%         |
|         | 6     | 0.1           | 0.95 | 0.01         | 51.40%         |
|         | 7     | 0.1           | 0.99 | 0.1          | 52.42%         |
|         | 8     | 0.1           | 0.99 | 0.001        | 66.33%         |
|         | 9     | 0.1           | 0.99 | 0.01         | 50.51%         |
|         | 10    | 0.01          | 0.9  | 0.1          | 74.56%         |
|         | 11    | 0.01          | 0.9  | 0.001        | 77.03%         |
|         | 12    | 0.01          | 0.9  | 0.01         | 63.44%         |
|         | 13    | 0.01          | 0.95 | 0.1          | 68.74%         |
|         | 14    | 0.01          | 0.95 | 0.001        | 77.89%         |
|         | 15    | 0.01          | 0.95 | 0.01         | 71.09%         |
|         | 16    | 0.01          | 0.99 | 0.1          | 75.51%         |
|         | 17    | 0.01          | 0.99 | 0.001        | 73.96%         |
|         | 18    | 0.01          | 0.99 | 0.01         | 63.36%         |
|         | 19    | 0.001         | 0.9  | 0.1          | 87.04%         |
|         | 20    | 0.001         | 0.9  | 0.001        | 86.85%         |
|         | 21    | 0.001         | 0.9  | 0.01         | 86.24%         |
|         | 22    | 0.001         | 0.95 | 0.1          | 87.71%         |
|         | 23    | 0.001         | 0.95 | 0.001        | 87.39%         |


# Justification and Conclusion
On varying multiple hyperparameters and optimizers, I got the following results:
❖ The Adagrad optimizer performed the best with the best top-5 accuracy = 88.46%
❖ Next to it was Adam optimizer with the best top-5 accuracy = 87.93%
❖ RMSprop performed the worst with the best top-5 accuracy = 87.71%
RMS prop performed significantly worse than the other 2 optimizers, furthermore, the result of
RMSprop was heavily influenced by the parameters chosen, the result can be explained by the fact
thatRMSprop does not include momentum, which can help smooth out the optimization process and
improve convergence. If the optimization landscape is particularly noisy or difficult to navigate,
RMSprop may not perform as well as other algorithms that include momentum, such as Adam or
Adagrad. RMSprop has several hyperparameters that need to be chosen carefully for optimal
performance, including the learning rate, decay rate, and epsilon. If these hyperparameters are
chosen poorly, RMSprop may not be able to find the optimal solution or may converge too slowly.
The possible reason Adagrad would have performed better is that it is easier to tune, Adagrad has
fewer hyperparameters to tune while Adam has several hyperparameters to tune, including the learning
rate, the decay rates for the first and second moments, and an epsilon value. This means that Adagrad
can be easier to tune, but it may not be as flexible as Adam for more complex problems.
It was further observed that the result of RMS prop was heavily influenced by the learning rate. I
received very low accuracy for learning rate that were higher, If the learning rate is too high, the
weight updates can become too large, causing the algorithm to overshoot the optimal weights and
diverge. a high learning rate can cause the RMSprop algorithm to perform poorly because it may
oscillate around the optimal weights, causing the optimization process to become unstable. This can
happen if the learning rate is too high relative to the size of the gradients. In such cases, the algorithm
may benefit from a smaller learning rate, which can lead to smoother updates and more stable
convergence

