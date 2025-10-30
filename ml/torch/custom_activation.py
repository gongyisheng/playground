# # Problem: Write a Custom Activation Function

# ### Problem Statement
# You are tasked with implementing a **custom activation function** in PyTorch that computes the following operation: 
 
# $ \text{activation}(x) = \tanh(x) + x $ 

# Once implemented, this custom activation function will be used in a simple linear regression model.

# ### Requirements
# 1. **Custom Activation Function**:
#    - Implement a class `CustomActivationModel` inheriting from `torch.nn.Module`.
#    - Define the `forward` method to compute the activation function \( \text{tanh}(x) + x \).

# 2. **Integration with Linear Regression**:
#    - Use the custom activation function in a simple linear regression model.
#    - The model should include:
#      - A single linear layer.
#      - The custom activation function applied to the output of the linear layer.

# ### Constraints
# - The custom activation function should not have any learnable parameters.
# - Ensure compatibility with PyTorch tensors for forward pass computations.

# <details>
#   <summary>💡 Hint</summary>
#   Some details: https://stackoverflow.com/questions/55765234/pytorch-custom-activation-functions
# </details>

import torch
import torch.nn as nn
import torch.optim as optim

def generate_data():
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10
    Y = 2 * X + 3 + torch.rand(100, 1)
    return X, Y

class CustomActivationModel(nn.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def custom_activation(self, x):
        return torch.tanh(x) + x

    def forward(self, x):
        return self.custom_activation(self.linear(x))

def train(epochs, lr):
    model = CustomActivationModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X, Y = generate_data()

    for epoch in range(epochs):
        predictions = model(X)
        loss = criterion(predictions, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model

def eval(model):
    w, b = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data
    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    model = train(1000, 0.01)
    eval(model)