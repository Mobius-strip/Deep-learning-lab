# Question 01 [75 marks] :

Let your date of birth be DD/MM/YY. Let the last three digits of your roll number at
IITJ is ABC. Finally, let your first name be FIRST and last name be LAST. And your program will
be PROG.
- Train Student-Teacher model with the following details.
Dataset details
● For B.Tech students:
○ CIFAR10 dataset if ABC is even; SVHN otherwise.
● For M.Tech/PhD students:
○ TinyImageNet dataset

Weight Initialization:
● Xavier initialization if MM is even; He initialization otherwise.

Pooling Operation Details:
● AvgPool if MM is even; MaxPool otherwise.
Model Details:
● For B.Tech students:
○ Teacher Network layers:
■ If last digit of ABC is less than 6:
● If ABC is even, your network should have 7 conv layers and 1 pool
layer.
● If ABC is odd, your network should have 8 conv layers and 1 pool
layers.

■ If last digit of ABC is greater than or equal to 6:
● Your network should have 10 conv layers and 1 pooling layer.
■ If ABC is even, your network will have 6 filters in the first layer.
■ If ABC is odd, your network will have 8 filters in the first layer.
○ Student Network:
■ If DD is even then student network is 2 conv layers + 1 pool layer
■ Else 3 conv layers + 1 pool layer
○ Fully-Connected layers (in both teacher and student networks):
■ If the sum of digits of ABC is even, your network should have 1 FC layer
with 512 nodes.

Minor - 2

Submission Deadline: 11:59 PM, 28th March 2023

■ If the sum of digits of ABC is odd, your network should have 1 FC layer
with 256 nodes.
● For M.Tech/PhD students:
○ Teacher Network layers:
■ If ABC is even, your network should have 10 conv layers and 2 pool
layers.
■ If ABC is odd, your network should have 12 conv layers and 1 pool layer.
■ If ABC is even, your network will have 8 filters in the first layer.
■ If ABC is odd, your network will have 12 filters in the first layer.
○ Student Network:
■ If DD is even then student network is 3 conv layers + 1 pool layer
■ Else 4 conv layers + 1 pool layer
○ Fully-Connected layers (in both teacher and student networks):
■ If the sum of digits of ABC is even, your network should have 1 FC layer
with 1024 nodes.
■ If the sum of digits of ABC is odd, your network should have 1 FC layer
with 512 nodes.

Report the performance of the student network and compare it with the teacher model. Also
compare the performance with and without EMA (Exponential Moving Average) updates.


# Question 02 [25+25 = 50 marks] :


Suppose your instructor is working on creating a question set for the DL course. Since we have
finished 2⁄3 of the DL course so far, your instructor wants your help in creating an interesting
question.
(a) Based on all the topics we have studied, design 3 interesting and challenging DL
questions (Min. 1 question numerical and Min. 1 question theoretical). Simple questions
and textbook questions are not allowed.
(b) Run ChatGPT and obtain the answers to your questions. Justify that the answer
produced by ChatGPT is correct.
