# Imports
import torch
import torch.nn as nn # all network module
import torch.optim as optim # all the optimizer algorithms
import torch.nn.functional as F # all the fuctions w/o parameters ex) LeRu
from torch.utils.data import DataLoader # easier data management
import torchvision.datasets as datasets # pytorch standard dataset
import torchvision.transforms as transforms # 
from customDataset import CatsAndDogsDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64 # 한번에 학습시킬 data들 묶음
num_epochs = 1

# Load Data
dataset = CatsAndDogsDataset(csv_file = 'cats_dogs.csv', root_dir = 'cats_dogs_resized', transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset= train_set, batch_size=,batch_size, shuffle=True)
test_loader = DataLoader(dataset= test_set, batch_size=,batch_size, shuffle=True)

# model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 모델의 parameters 

# Train Netowrk
for epoch in range(num_epochs):
    for batch, (data, targets) in enumerate(train_loader): # data is images, targets are correct digits for each image
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape) # [64, 1, 28, 28] => [Num_of_images, black_or_white, size_of_images]
        
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

            scores = model(x) # 64x10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)