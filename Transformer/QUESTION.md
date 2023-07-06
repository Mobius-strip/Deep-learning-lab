# Question 1 [50 marks]

Perform image classification on selected classes[check below in Note] of CIFAR-10 dataset
using transformer model with following variations:

1. Use cosine positional embedding with six encoders and decoder layers with eight
heads. Use relu activation in the intermediate layers. Marks [20]
2. Use learnable positional encoding with four encoder and decoder layers with six heads.
Use relu activation in the intermediate layers. Marks [20]
3. For parts (a) and (b) change the activation function in the intermediate layer from relu to
tanh and compare the performance. Marks [10]

Note: Those who have
Even roll number : select odd classes
Odd roll number: select even classes
Reference Blog:
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide
-96c3313c2e0c
https://theaisummer.com/positional-embeddings/

# Questions 2 [50 marks]

Based on the lecture by Dr. Anush on DLOPs, you have to perform the following experiments :

● Load and preprocessing CIFAR10 dataset using standard augmentation and
normalization techniques [10 Marks]

● Train the following models for profiling them using during the training step [5 *2 =
10Marks]

○ Conv -> Conv -> Maxpool (2,2) -> Conv -> Maxpool(2,2) -> Conv -> Maxpool(2,2)
■ You can decide the parameters of convolution layers and activations on
your own.
■ Make sure to keep 4 conv-layers and 3 max-pool layers in the order
describes above.

○ VGG16
● After the profiling of your model, figure out the minimum change in the architecture that
would lead to a gain in performance and decrease training time on CIFAR10 dataset as
compared to one achieved before. [30 Marks]
