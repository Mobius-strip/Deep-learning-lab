# -*- coding: utf-8 -*-
"""b20bb030_DLOps_Assignment-3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sRM2yOFx4AxTZgPWr9BewHmNfMe9I-wp
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



# Define the transform to apply to the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 training set, only including the specified classes
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

def filter(train_dataset) : 
      

      new_train_target=[]
      for label in train_dataset.targets:
        if(label==1):
          new_train_target.append(0)
        if(label==3):
          new_train_target.append(1)

        if(label==5):
          new_train_target.append(2)

        if(label==7):
         
          new_train_target.append(3)

        if(label==9):

          new_train_target.append(4)
          
        if(label==0):
          
          new_train_target.append(20)
        if(label==2):
         
          new_train_target.append(21)

        if(label==4):
         
          new_train_target.append(22)

        if(label==6):
        
          new_train_target.append(23)

        if(label==8):
         
          new_train_target.append(24)


        train_dataset.targets=new_train_target  

      return train_dataset

train_dataset = filter(trainset)
test_dataset = filter(testset)

def return_indices(train_set):

    indices=[]
    l=[0,1,2,3,4]
    for i in range(len(train_set.targets)):
      if(train_set.targets[i] in l ):
        indices.append(i)
    return indices

data2=[]
target2=[]
indices=return_indices(train_dataset)
for index in indices:
  target2.append(train_dataset.targets[index])
  data2.append(train_dataset.data[index])

train_dataset.data=[]
train_dataset.target=[]

train_dataset.data=data2
train_dataset.target=target2

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

train_dataset = MyDataset(data=data2, targets=target2, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=512, shuffle=True)

data3=[]
target3=[]
indices=return_indices(test_dataset)
for index in indices:
  target2.append(test_dataset.targets[index])
  data2.append(test_dataset.data[index])

test_dataset = MyDataset(data=data3, targets=target3, transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset , batch_size=512, shuffle=True)

import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
    
import math

class AbsPosEmb1D(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        emb = torch.zeros(1, seq_len, hidden_dim)

        emb[0, :, 0::2] = torch.sin(position * div_term)
        emb[0, :, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Parameter(emb, requires_grad=True)

    def forward(self, x):
        return x + self.emb[:, :x.size(1)]

class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.pos_emb = AbsPosEmb1D(n_patches ** 2 + 1, hidden_d)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Tanh(),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Applying positional embedding
        tokens = self.pos_emb(tokens)
        
        # Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens)
            
        # Getting the classification token only
        tokens = tokens[:, 0]
        
        return self.mlp(tokens) # Map to output dimension, output categor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
model = MyViT((3, 32, 32), n_patches=4, n_blocks=4, hidden_d=6, n_heads=6, out_d=10).to(device)
N_EPOCHS = 5
LR = 0.005

# Training loop
train_losses=[]
test_losses = []
train_acc = []
val_acc = []
optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(y_hat, 1)
        total_correct += (predicted == y).sum().item()

        total_samples += y.size(0)

    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
    train_losses.append(train_loss)
    train_acc.append(100 * total_correct/total_samples)
# Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        val_acc.append(correct / total * 100)
        test_losses.append(test_loss) 


plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='testing loss')
plt.legend()
plt.savefig("loss curve for tanh q1_2" )
plt.close()
plt.plot(train_acc, label='training acc')
plt.plot(val_acc, label='val acc')
plt.legend()
plt.savefig("ACCURACY curve for tanh q1_2" )
plt.close()


import matplotlib.pyplot as plt

# Instantiate your MyViT model
my_model = MyViT((3, 32, 32), n_patches=4, n_blocks=4, hidden_d=6, n_heads=6, out_d=10).to(device)

# Extract the positional embeddings
pos_emb = my_model.pos_emb.emb.squeeze().cpu().detach().numpy()

# Plot the positional embeddings as a heatmap
plt.imshow(pos_emb, cmap='viridis', aspect='equal')
plt.colorbar()
plt.savefig("Heatmap for tanh q1_2" )
plt.show()