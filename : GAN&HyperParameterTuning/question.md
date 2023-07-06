# Question 1 [50 marks]
Train a Conditional Deep Convolutional Generative Adversarial Network (cDCGAN) on Dataset.
(You may use this for Reference) [25 Marks]
A. Generate 50 Samples of each class from your trained generator. [5 Marks]
B. Train a ResNet18 classifier on the given dataset and treat the generated samples as test
dataset and report following [20 Marks]
1. F1 Score for each class
2. Confusion matrix

Reference: https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/

# Questions 2 [50 marks]

Train a CNN based classification model and perform Optimized Hyperparameter Tuning using
Optuna Library on the below-mentioned dataset. Perform 100 trials.
Hyperparameters should be
1) No of Convolution Layers 3 to 6
2) Number of Epochs 10 to 50
3) Learning rate 0.0001 to 0.1
Report the observations and the best trial. Report how many trials were pruned
For Even Roll Number MNIST
For Odd Roll Number Fashion MNIST

Reference: https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN
