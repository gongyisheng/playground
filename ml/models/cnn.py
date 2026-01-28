# # Problem: Implement a CNN for CIFAR-10 in PyTorch

# ### Problem Statement
# You are tasked with implementing a **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10** dataset using PyTorch. The model should contain convolutional layers for feature extraction, pooling layers for downsampling, and fully connected layers for classification. Your goal is to complete the CNN model by defining the necessary layers and implementing the forward pass.

# ### Requirements
# 1. **Define the CNN Model**:
#    - Add **convolutional layers** for feature extraction.
#    - Add **pooling layers** to reduce the spatial dimensions.
#    - Add **fully connected layers** to output class predictions.
#    - The model should be capable of processing input images of size `(32x32x3)` as in the CIFAR-10 dataset.

# ### Constraints
# - The CNN should be designed with multiple convolutional and pooling layers followed by fully connected layers.
# - Ensure the model is compatible with the CIFAR-10 dataset, which contains 10 classes.


# <details>
#   <summary>💡 Hint</summary>
#   Add the convolutional (conv1, conv2), pooling (pool), and fully connected layers (fc1, fc2) in CNNModel.__init__.
#   <br>
#   Implement the forward pass to process inputs through these layers.
# </details>

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def load_dataset():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, dataset, device, epochs, lr, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Training loop
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

def eval(model, dataset, device, batch_size):
    # Evaluate on the test set
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    model = CNNModel()
    train_dataset, test_dataset = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    train(model, train_dataset, device, 10, 0.001, 64)
    eval(model, test_dataset, device, 64)

    # output:
    # Epoch [1/10], Loss: 1.5841
    # Epoch [2/10], Loss: 0.4168
    # Epoch [3/10], Loss: 0.6398
    # Epoch [4/10], Loss: 0.4325
    # Epoch [5/10], Loss: 0.1943
    # Epoch [6/10], Loss: 0.1350
    # Epoch [7/10], Loss: 0.0374
    # Epoch [8/10], Loss: 0.1173
    # Epoch [9/10], Loss: 0.0848
    # Epoch [10/10], Loss: 0.0404
    # Test Accuracy: 67.95%