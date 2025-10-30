# # Problem: Implement Custom Loss Function (Huber Loss)

# ### Problem Statement
# You are tasked with implementing the **Huber Loss** as a custom loss function in PyTorch. The Huber loss is a robust loss function used in regression tasks, less sensitive to outliers than Mean Squared Error (MSE). It transitions between L2 loss (squared error) and L1 loss (absolute error) based on a threshold parameter $ \delta $.

# The Huber loss is mathematically defined as:
# $$
# L_{\delta}(y, \hat{y}) = 
# \begin{cases} 
# \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta, \\
# \delta \cdot (|y - \hat{y}| - \frac{1}{2} \delta) & \text{for } |y - \hat{y}| > \delta,
# \end{cases}
# $$

# where:
# - $y$ is the true value,
# - $\hat{y}$ is the predicted value,
# - $\delta$ is a threshold parameter that controls the transition between L1 and L2 loss.

# ### Requirements
# 1. **Custom Loss Function**:
#    - Implement a class `HuberLoss` inheriting from `torch.nn.Module`.
#    - Define the `forward` method to compute the Huber loss as per the formula.

# 2. **Usage in a Regression Model**:
#    - Integrate the custom loss function into a regression training pipeline.
#    - Use it to compute and optimize the loss during model training.

# ### Constraints
# - The implementation must handle both scalar and batch inputs for $ y $ (true values) and $ \hat{y} $ (predicted values).


# Extra Details: https://en.wikipedia.org/wiki/Huber_loss

# <details>
#   <summary>💡 Hint</summary>
#   Some details: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
# </details>

import torch
import torch.nn as nn
import torch.optim as optim


def generate_data():
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  
    Y = 2 * X + 3 + torch.randn(100, 1)
    return X, Y

class HubLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HubLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = torch.abs(y_pred - y_true)
        loss = torch.where(error <= self.delta, 
                           0.5 * error**2, # L2 loss for small errors
                           self.delta * (error - 0.5 * self.delta)) # L1 loss for large errors
        return loss.mean() 

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1) 

    def forward(self, x):
        return self.linear(x)

def train(epochs, lr, delta=1.0):
    model = LinearRegressionModel()
    criterion = HubLoss(delta=delta)
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
    model = train(1000, 0.01, 2)
    eval(model)

    # delta = 1
    # Epoch [100/1000], Loss: 0.6620
    # Epoch [200/1000], Loss: 0.5847
    # Epoch [300/1000], Loss: 0.5197
    # Epoch [400/1000], Loss: 0.4672
    # Epoch [500/1000], Loss: 0.4254
    # Epoch [600/1000], Loss: 0.3929
    # Epoch [700/1000], Loss: 0.3677
    # Epoch [800/1000], Loss: 0.3481
    # Epoch [900/1000], Loss: 0.3332
    # Epoch [1000/1000], Loss: 0.3220
    # Learned weight: 2.0317, Learned bias: 2.7418
    # Predictions for [[4.0], [7.0]]: [[10.86854362487793], [16.963638305664062]]

    # delta = 2
    # Epoch [100/1000], Loss: 0.7559
    # Epoch [200/1000], Loss: 0.6057
    # Epoch [300/1000], Loss: 0.5045
    # Epoch [400/1000], Loss: 0.4398
    # Epoch [500/1000], Loss: 0.3992
    # Epoch [600/1000], Loss: 0.3736
    # Epoch [700/1000], Loss: 0.3575
    # Epoch [800/1000], Loss: 0.3474
    # Epoch [900/1000], Loss: 0.3411
    # Epoch [1000/1000], Loss: 0.3371
    # Learned weight: 1.9891, Learned bias: 2.9994
    # Predictions for [[4.0], [7.0]]: [[10.955925941467285], [16.923311233520508]
