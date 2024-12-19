import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(torch.sigmoid(self.conv1(x))) #convolution+activation+pooling
        x = self.pool(torch.sigmoid(self.conv2(x)))

        x = x.view(-1,16*5*5) #flattening

        x = torch.sigmoid(self.fc1(x)) #use F.tanh or F.relu instead of torch.sigmoid
        x = torch.sigmoid(self.fc2(x))

        x = self.fc3(x)

        return x

# Transforming
transform = transforms.Compose([transforms.Resize((32,32)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,),(0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initializing
model = LeNet5()
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Evaluation function
def evaluate(model, test_loader):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():  
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')


# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)

# Evaluate the model
evaluate(model, test_loader)
