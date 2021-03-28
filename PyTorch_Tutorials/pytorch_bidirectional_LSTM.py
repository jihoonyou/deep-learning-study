# Imports
import torch
import torch.nn as nn # all network module
import torch.optim as optim # all the optimizer algorithms
import torch.nn.functional as F # all the fuctions w/o parameters ex) LeRu
from torch.utils.data import DataLoader # easier data management
import torchvision.datasets as datasets # pytorch standard dataset
import torchvision.transforms as transforms # 


 # Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64 # 한번에 학습시킬 data들 묶음
num_epochs = 1

# Create a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, input, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train = True, transform=transforms.ToTensor(), download=True) # download data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train = False, transform=transforms.ToTensor(), download=True) # download data
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 모델의 parameters 

# Train Netowrk
for epoch in range(num_epochs):
    for batch, (data, targets) in enumerate(train_loader): # data is images, targets are correct digits for each image
        data = data.to(device=device).squeeze(1)
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x) # 64x10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)