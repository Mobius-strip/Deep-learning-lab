# Question 1 [80 marks]
## Train ResNet18 on Tiny ImageNet dataset (download from here) with X as the optimizer for classification task. Plot curves for training loss, training accuracy, validation accuracy and report the final test accuracy. Here consider accuracy as top-5 accuracy.
1. Use CrossEntropy as the final classification loss function [10 marks]
2. Use Triplet Loss with hard mining as the final classification loss function [30 marks]
3. Use Central Loss as the final classification loss function [40 marks]
Compare the performance of different models and analyze the results in the report.

Note - The code for ResNet18 architecture and the loss functions needs to be implemented
from scratch. Directly importing from the library is not allowed and 0 marks will be awarded for
that.
X = Adam, if last digit of your roll no. is odd
X = SGD, if last digit of your roll no. is even


# Question 2 [20 marks]

## Implement a multi-layered classifier where weights of each layer is calculated greedily using layer-wise pretraining with the help of auto-encoders on STL-10 dataset. Train a classifier having X structure (excluding input and output layers) for classification task on the test set.

1. Report the classification accuracy on the test set and plot loss curves on the training and
evaluation set.
2. Report the class-wise accuracy of each class.
3. Plot t-sne for this model (use embeddings from layer X[3]) . Use the first 500 images of
each class from the test dataset for this visualization.
X = [1024,1200,728,512,128], if last digit of your roll no. is odd
X = [1024,1000,500,256, 128,64], if last digit of your roll no. is even
