# Imports
import torch
import torch.nn as nn # all network module
import torch.optim as optim # all the optimizer algorithms
import torch.nn.functional as F # all the fuctions w/o parameters ex) LeRu
from torch.utils.data import DataLoader # easier data management
import torchvision.datasets as datasets # pytorch standard dataset
import torchvision.transforms as transforms # 

# Create Fully Conneceted Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # MINST dataset 28x28 = 784
        super(NN, self).__init__() # call parent's init
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size =784
num_classes = 10
learning_rate = 0.001
batch_size = 64 # 한번에 학습시킬 data들 묶음
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train = True, transform=transforms.ToTensor(), download=True) # download data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train = False, transform=transforms.ToTensor(), download=True) # download data
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 모델의 parameters 

# Train Netowrk
for epoch in range(num_epochs):
    for batch, (data, targets) in enumerate(train_loader): # data is images, targets are correct digits for each image
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape) # [64, 1, 28, 28] => [Num_of_images, black_or_white, size_of_images]
        
        # Get to correct shape
        data = data.reshape(data.shape[0], -1) # remain first dimension to remiann 64 [64, 784]
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accruacy on training & test to see hwo good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval() 

    with torch.no_grad(): #unnecessary computation / no gradient calculation
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x) # 64x10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)