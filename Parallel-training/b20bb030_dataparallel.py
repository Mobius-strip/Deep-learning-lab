import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import psutil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import time

# Define device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])
from torch.utils.data import random_split

# Load train and test datasets
dataset = datasets.SVHN(root='data/', download=True, transform=transform)

val_size = 12000
train_size = len(dataset) - val_size
train_ds, test_ds = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=64, shuffle=True)
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x


model = ResNet18(num_classes=10)    

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.4)



# Define lists to store accuracy, loss and time taken per epoch
accuracy_list = []
loss_list = []
time_list = []
memory_list = []

model=model.to(device)

model = nn.DataParallel(model) 

# Train the model
for epoch in range(30):  # loop over the dataset multiple times
    
    start_time = time.time()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * total_correct / total_samples
    epoch_time = time.time() - start_time

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)
    time_list.append(epoch_time)
    memory_list.append(psutil.Process().memory_info().rss / 1024 / 1024)

    print('Epoch %d: loss=%.3f, accuracy=%.2f%%, time=%.2fs' % (epoch+1, epoch_loss, epoch_accuracy, epoch_time))
    print(psutil.Process().memory_info().rss / 1024 / 1024)

print('Finished Training')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print('Accuracy of the network on the 10000 test images: %.2f%%' % test_accuracy)


import matplotlib.pyplot as plt


# Plot and save the accuracy list
plt.figure(figsize=(10,5))
plt.plot(accuracy_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()
plt.close()

# Plot and save the loss list
plt.figure(figsize=(10,5))
plt.plot(loss_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
plt.close()

# Plot and save the time list
plt.figure(figsize=(10,5))
plt.plot(time_list, label='Time')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.title('Time over epochs')
plt.legend()
plt.savefig('time_plot.png')
plt.show()
plt.close()


# Plot and save the accuracy list
plt.figure(figsize=(10,5))
plt.plot(memory_list, label='mem')
plt.xlabel('Epoch')
plt.ylabel('mem')
plt.title('mem over epochs')
plt.legend()
plt.savefig('mem_plot.png')
plt.show()
plt.close()


