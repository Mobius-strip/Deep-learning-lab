## Q1. In this assignment you are required to train a Convolutional Neural Network on two datasets using slurm and submit the jobs [100 marks]
a. If the last digit of your roll number is even:
Train ResNet-18 on FashionMNIST dataset [10 marks]
b. If the last digit of your roll number is odd:
Train MobileNet_V2 on CIFAR-10 dataset [10 marks]
● Download the datasets and extract them to folders on the GPU server and preprocess
the data. [10 marks]
(Use default PyTorch dataloader function mentioning the dataset path and utilize
different transforms for preprocessing [compulsory])
● Set the loss function, optimiser, and metrics and compile the model [30 marks]
● Use Slurm to submit a job for training the model on the GPU server [15 marks]
● Also measure the training time (use timeit module for instance) for 10 and 15 epochs [10
marks]
● Try to identify the set of hyperparameters that results in similar performance as
compared to the best performance in the previous step but with lower training time [25
marks]
