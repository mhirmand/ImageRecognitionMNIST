import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: 1 input channel, 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        # ReLU activation
        self.relu = nn.ReLU()
        # MaxPool: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # FC Layer: 16*13*13 inputs -> 10 outputs
        self.fc = nn.Linear(16*13*13, 10)
        
    def forward(self, x):
        x = self.conv1(x)       # Output: (16, 26, 26)
        x = self.relu(x)        # Non-linear activation
        x = self.pool(x)        # Output: (16, 13, 13)
        x = x.view(-1, 16*13*13) # Flatten for FC
        x = self.fc(x)          # Final classification
        return x

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# print train and test sizes.
print(f'Train size: {len(trainset)}')
print(f'Test size: {len(testset)}')

# Initialize the network, loss function and optimizer
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training the network
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 9:    # print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Train loss: {running_loss / 100:.3f} ')
            running_loss = 0.0

print('Finished Training')

# Testing the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')