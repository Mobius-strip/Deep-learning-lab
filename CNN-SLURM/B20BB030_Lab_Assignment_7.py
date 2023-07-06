import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import timeit
import matplotlib.pyplot as plt
import pickle
print("File running")
# Define the transform to convert grayscale to fake RGB
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.stack([x.squeeze()]*3, 0)),
  
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.array(self.data.iloc[idx, 1:], dtype=np.uint8).reshape((28, 28, 1))
        label = np.array(self.data.iloc[idx, 0], dtype=np.int64)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load the Fashion-MNIST dataset
train_dataset = FashionMNISTDataset('/iitjhome/b20bb030/scratch/data/b20bb030/lab7/archive/fashion-mnist_test.csv', transform=transform)
test_dataset = FashionMNISTDataset('/iitjhome/b20bb030/scratch/data/b20bb030/lab7/archive/fashion-mnist_train.csv', transform=transform)

# Create data loaders




# Define the ResNet-18 model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False, num_classes=10)
model=model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Train the model
# Set hyperparameters
num_epochs_list = [5, 10, 15]
learning_rate_list = [0.001, 0.01, 0.1]
batch_size_list = [64, 128, 256]
result=[]
for num_epochs in num_epochs_list:

    for learning_rate in learning_rate_list:
        for batch_size in batch_size_list:
            # Train the model
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            train_losses=[]
            test_losses = []
            train_acc = []
            val_acc = []
            start_time = timeit.default_timer()
            for epoch in range(num_epochs):
                r=0.0
                total_correct = 0
                total_samples = 0
                model = model.train()

                for images, labels in trainloader:
                    # Move the data to the GPU if available
                    if torch.cuda.is_available():
                        images, labels = images.to(device), labels.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(images)

                    # Compute the loss
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Update the weights
                    optimizer.step()
                    r=r+loss.item()
                    r /= len(trainloader)

                    # _, predicted = torch.max(outputs.data, 1)
                    # total_train += outputs.size(0)
                    # correct_train += (predicted == outputs).sum().item()

                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()

                    total_samples += labels.size(0)

                train_losses.append(r) 
                train_acc.append(100 * total_correct/total_samples)
                # Print the loss and accuracy for the current epoch
                with torch.no_grad():
                    total_loss = 0.0
                    total_correct = 0
                    total_samples = 0
                    for images, labels in testloader:
                        # Move the data to the GPU if available
                        if torch.cuda.is_available():
                            images, labels = images.to(device), labels.to(device)

                        # Forward pass
                        outputs = model(images)

                        # Compute the loss
                        loss = criterion(outputs, labels)
                        total_loss += loss.item() * images.size(0)

                        # Compute the accuracy
                        _, predicted = torch.max(outputs, 1)
                        total_correct += (predicted == labels).sum().item()
                        total_samples += labels.size(0)

                    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch+1, num_epochs, total_loss/len(testloader.dataset),
                                  100.0 * total_correct / len(testloader.dataset)))
                val_acc.append(100 * total_correct/ total_samples)
                total_loss /= len(testloader)
                test_losses.append(total_loss) 
            elapsed_time = timeit.default_timer() - start_time

            print(elapsed_time)
            print(num_epochs)
            print(learning_rate)
            print(batch_size)

            result_i=[]
            result_i.append(elapsed_time)
            result_i.append(num_epochs)
            result_i.append(learning_rate)
            result_i.append(batch_size)
            result_i.append(val_acc[-1])
            result.append(result_i)

            plt.plot(train_losses, label='training loss')
            plt.plot(test_losses, label='testing loss')
            plt.legend()
            plt.savefig("loss curve for  {} , {}, {}, {},{}.png".format(num_epochs, learning_rate, batch_size,val_acc[-1],elapsed_time))
            plt.close()
            plt.plot(train_acc, label='training acc')
            plt.plot(val_acc, label='val acc')
            plt.legend()
            plt.savefig("ACCURACY curve for  {} , {}, {},{}.{}.png".format(num_epochs, learning_rate, batch_size,val_acc[-1],elapsed_time))
            plt.close()
            print(f"Elapsed time: {elapsed_time:.2f} seconds for num_epochs={num_epochs}, learning_rate={learning_rate}, batch_size={batch_size}")




with open('my_list.pickle', 'wb') as f:
    # serialize the list and write it to the file
    pickle.dump(result, f)


