# Question 01: [10 marks + 30 marks bonus]

## Perform Attribute Prediction Task on CelebA [1] dataset for the attributes available in the dataset. For this task, use a VGG16/ResNet18 backbone and train your model in a Multi-TaskLearning fashion. Report the following:

1. Train the model for the prediction of 8 attributes out of 40 attributes and report the
following:
a. Mention your choice of attributes, backbone, and other parameters and the
reason behind your choice. [2 marks]
b. Report task-wise accuracy. [4 marks]
c. Report the overall accuracy. [4 marks]


# [Bonus] Refer to the work of Malhotra et al. [2] and report the following:

a. Drop rate for each task (any of the four metrics described
in the paper). [10 marks]

b. Explanation step-by-step on how you computed the drop rate. [5 marks]

c. Analyze your observations and discuss them in the report. [5 marks]
d. Use your drop rate to calculate task-wise activation probability (since you are
using a single metric to calculate drop rate, task-wise activation probability can
be modified accordingly) and implement the DST algorithm and report the gain in
performance with your analysis. [10 marks]

# References:
[1] Liu, Ziwei, Ping Luo, Xiaogang Wang, and Xiaoou Tang. "Large-scale celebfaces attributes
(celeba) dataset." Retrieved August 15, no. 2018 (2018): 11.
[2] Malhotra, Aakarsh, Mayank Vatsa, and Richa Singh. "Dropped Scheduled Task: Mitigating
Negative Transfer in Multi-task Learning using Dynamic Task Dropping." Transactions on
Machine Learning Research.
