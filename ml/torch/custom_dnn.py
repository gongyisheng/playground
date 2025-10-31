# # Problem: Implement a Deep Neural Network

# ### Problem Statement
# You are tasked with constructing a **Deep Neural Network (DNN)** model to solve a regression task using PyTorch. The objective is to predict target values from synthetic data exhibiting a non-linear relationship.

# ### Requirements
# Implement the `DNNModel` class that satisfies the following criteria:

# 1. **Model Definition**:
#    - The model should have:
#      - An **input layer** connected to a **hidden layer**.
#      - A **ReLU activation function** for non-linearity.
#      - An **output layer** with a single unit for regression.

# <details> 
#   <summary>💡 Hint</summary> 
#   - Use `nn.Sequential` to simplify the implementation of the `DNNModel`. 
#   - Experiment with different numbers of layers and hidden units to optimize performance. 
#   - Ensure the final layer has a single output unit (since it's a regression task). 
# </details> 
# <details> 
#   <summary>💡 Bonus: Try Custom Loss Functions</summary> 
#   Experiment with custom loss functions (e.g., Huber Loss) and compare their performance with MSE. 
# </details>

import torch
import torch.nn as nn
import torch.optim as optim

def generate_data():
    X = torch.rand(100, 2) * 10
    Y = (X[:, 0] + X[:, 1] * 2).unsqueeze(-1) + torch.rand(100, 1)
    return X, Y

class CustomDNNModel(nn.Module):
    def __init__(self):
        super(CustomDNNModel, self).__init__()
        self.fc1 = nn.Linear(2, 10) # input to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1) # hidden layer to output
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def train(model, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X, Y = generate_data()

    for epoch in range(epochs):
        predictions = model(X)
        loss = criterion(predictions, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def eval(model):
    # Testing on new data
    X_test = torch.tensor([[4.0, 3.0], [7.0, 7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    model = CustomDNNModel()
    train(model, 1000, 0.01)
    eval(model)

    # output
    # Epoch [100/1000], Loss: 0.2265
    # Epoch [200/1000], Loss: 0.1333
    # Epoch [300/1000], Loss: 0.1051
    # Epoch [400/1000], Loss: 0.0878
    # Epoch [500/1000], Loss: 0.0791
    # Epoch [600/1000], Loss: 0.0753
    # Epoch [700/1000], Loss: 0.0739
    # Epoch [800/1000], Loss: 0.0734
    # Epoch [900/1000], Loss: 0.0733
    # Epoch [1000/1000], Loss: 0.0732
    # Predictions for [[4.0, 3.0], [7.0, 7.0]]: [[10.530770301818848], [21.563879013061523]]