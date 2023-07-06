# Question 01 [50 marks] :
Let your date of birth be DD/MM/YY. Let the last three digits of your roll number at
IITJ is ABC. Finally, let your first name be FIRST and last name be LAST. And your program will
be PROG.
- Train a CNN with the following details.
Dataset details
● For B.Tech students:
○ FashionMNIST dataset if ABC is even; MNIST otherwise.
● For M.Tech/PhD students:
○ CIFAR10 dataset

Weight Initialization:
● Xavier initialization if MM is even; He initialization otherwise.
Data Augmentation Details:
● If DD is even, use flip and random noise augmentation.
● If DD is odd, rotate by 10 degrees and gaussian noise.
Pooling Operation Details:
● AvgPool if MM is even; MaxPool otherwise.
Target Classification Details:
● If the sum of DD, MM, and YY is even then your target 5-classes will be 0,2,4,6,8.
● If the sum of DD, MM, and YY is odd then your target 5-classes will be 1,3,5,7,9.
Model Details:
● For B.Tech students:
○ Feature Extraction layers:
■ If last digit of ABC is less than 6:
● If ABC is even, your network should have 4 conv layers and 1 pool
layer.
● If ABC is odd, your network should have 3 conv layers and 2 pool
layers.

■ If last digit of ABC is greater than or equal to 6:
● Your network should have 5 conv layers and 1 pooling layer.
■ If ABC is even, your network will have 10 filters in the first layer.
■ If ABC is odd, your network will have 8 filters in the first layer.
○ Fully-Connected layers:

Minor - 1

Submission Deadline: 11:59 PM, 13th February 2023

■ If the sum of digits of ABC is even, your network should have 1 FC layer
with 512 nodes.
■ If the sum of digits of ABC is odd, your network should have 1 FC layer
with 256 nodes.
● For M.Tech/PhD students:
○ Feature Extraction layers:
■ If ABC is even, your network should have 5 conv layers and 2 pool layers.
■ If ABC is odd, your network should have 6 conv layers and 1 pool layer.
■ If ABC is even, your network will have 16 filters in the first layer.
■ If ABC is odd, your network will have 12 filters in the first layer.
○ Fully-Connected layers:
■ If the sum of digits of ABC is even, your network should have 1 FC layer
with 1024 nodes.
■ If the sum of digits of ABC is odd, your network should have 1 FC layer
with 512 nodes.

# Question 02 [50 marks] :
Train an autoencoder with the same details as given before and compare your results with CNN
results. The number of AE layers will be 4 if ABC even else 3. The classification layer will be
single FC with 512 nodes if your MM is even, else 256. Rest of the hyperparameters are left to
you (please list them and explain the reason of your choice).
