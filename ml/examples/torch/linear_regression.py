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

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_data():
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    Y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise
    return X, Y

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self, x):
        return self.linear(x)

def train(model, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X, Y = generate_data()
    model.train()

    for epoch in range(epochs):
        # Forward pass
        prediction = model(X)
        loss = criterion(prediction, Y)

        # Backward pass and optimization
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # Log epoch
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def train_with_loss_accumulation(model, epochs, lr, accumulation_steps=1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X, Y = generate_data()
    X.to(device)
    Y.to(device)
    optimizer.zero_grad()
    model.train()
    running_loss = 0

    for epoch in range(epochs):

        output = model(X)
        loss = criterion(output, Y)
        loss = loss / accumulation_steps
        running_loss += loss.item()

        loss.backward() # calculate gradient, store in model.weight.grad and model.bias.grad

        if (epoch+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}')
            running_loss = 0

def eval(model):
    model.eval()
    # Display the learned parameters
    [w, b] = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data
    X_test = torch.tensor([[4.0], [7.0]]).to(device)
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    model = LinearRegressionModel()
    train(model, 1000, 0.01)
    # train_with_loss_accumulation(model, 1000, 0.01, 2)
    eval(model)

    # Output:
    # Epoch [100/1000], Loss: 1.9016
    # Epoch [200/1000], Loss: 1.4304
    # Epoch [300/1000], Loss: 1.1384
    # Epoch [400/1000], Loss: 0.9576
    # Epoch [500/1000], Loss: 0.8456
    # Epoch [600/1000], Loss: 0.7762
    # Epoch [700/1000], Loss: 0.7332
    # Epoch [800/1000], Loss: 0.7065
    # Epoch [900/1000], Loss: 0.6900
    # Epoch [1000/1000], Loss: 0.6798
    # Learned weight: 1.9922, Learned bias: 2.9709
    # Predictions for [[4.0], [7.0]]: [[10.939563751220703], [16.916061401367188]]