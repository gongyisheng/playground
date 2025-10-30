# Problem Statement
# Your task is to implement a Linear Regression model using PyTorch. 
# The model should predict a continuous target variable based on a given set of input features.

# Requirements
#   Model Definition:
#       Implement a class LinearRegressionModel with:
#           A single linear layer mapping input features to the target variable.
#   Forward Method:
#       Implement the forward method to compute predictions given input data.

import torch
import torch.nn as nn
import torch.optim as optim

def generate_data():
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    Y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise
    return X, Y

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Define a linear layer that maps input features to the target variable
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # Compute the output by passing the input through the linear layer
        out = self.linear(x)
        return out

def train():
    # Initialize the model, loss function, and optimizer
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    X, Y = generate_data()

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')