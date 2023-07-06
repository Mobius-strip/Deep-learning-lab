# Question 1 [35 marks]
## Train a ResNet18 model on SVHN dataset using DataParallelism and DistributedDataParallelism for 30 epochs. You can use any library of your choice for parallel training of your model in PyTorch. [35 marks]
A. Use a batch size of 32 and train on a single GPU node. [5 Marks]
B. Use a batch size of 64 and train and two GPU nodes first using DataParallelism and then
using DistributedDataParallelism. [15 * 2 = 30 Marks]

1. Report all the hyper-parameters used for Parallel Training.
2. Compare the time (in seconds) and report the speed up. Show this speed up
using a graphical representation.
3. Describe your observations in terms of memory usage for multi-node training.

## Instructions to use multiple GPUs on server 172.25.0.15
Please follow the following instructions to enable multiple GPUs in the Slurm environment on
the server:
1. Navigate to the target directory and edit the batch script file.
2. Using vim or nano, edit line from “#SBATCH --gres=gpu” to “#SBATCH --gres=gpu:2”
(line 7 in test-gpu.sh).
3. Save this file and submit your job using sbatch command. This will allocate two GPUs for
your job.
4. You can monitor the memory usage of each GPU using nvidia-smi command.
