# # Problem: Write a custom Dataset and Dataloader to load from a CSV file

# ### Problem Statement
# You are tasked with creating a **custom Dataset** and **Dataloader** in PyTorch to load data from a given `data.csv` file. The loaded data will be used to run a pre-implemented linear regression model.

# ### Requirements
# 1. **Dataset Class**:
#    - Implement a class `CustomDataset` that:
#      - Reads data from a provided `data.csv` file.
#      - Stores the features (X) and target values (Y) separately.
#      - Implements PyTorch's `__len__` and `__getitem__` methods for indexing.

# 2. **Dataloader**:
#    - Use PyTorch's `DataLoader` to create an iterable for batch loading the dataset.
#    - Support user-defined batch sizes and shuffling of the data.

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def generate_data(output_path):
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    Y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

    # Save the generated data to data.csv
    data = torch.cat((X, Y), dim=1)
    df = pd.DataFrame(data.numpy(), columns=['X', 'Y'])
    df.to_csv(output_path, index=False)

class LinearRegressionDataset(Dataset):
    def __init__(self, csv_file):
        super(LinearRegressionDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.x = torch.tensor(self.data['X'].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(self.data['Y'].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self, x):
        return self.linear(x)

def train(model, dataset, epochs, lr, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log progress every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model

def eval(model):
    # Display the learned parameters
    [w, b] = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data
    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    csv_file_path = "/tmp/_data.csv"
    generate_data(csv_file_path)
    model = LinearRegressionModel()
    dataset = LinearRegressionDataset(csv_file_path)
    train(model, dataset, 1000, 0.01, 128)
    eval(model)

    # batch_size = 32 (not stable)
    # Epoch [100/1000], Loss: 1.5655
    # Epoch [200/1000], Loss: 0.4624
    # Epoch [300/1000], Loss: 1.4614
    # Epoch [400/1000], Loss: 0.3983
    # Epoch [500/1000], Loss: 0.5415
    # Epoch [600/1000], Loss: 1.6767
    # Epoch [700/1000], Loss: 1.0075
    # Epoch [800/1000], Loss: 0.3245
    # Epoch [900/1000], Loss: 0.6541
    # Epoch [1000/1000], Loss: 1.6099
    # Learned weight: 1.9207, Learned bias: 3.2333
    # Predictions for [[4.0], [7.0]]: [[10.91616153717041], [16.678312301635742]]

    # batch_size = 128 (stable)
    # Epoch [100/1000], Loss: 1.6039
    # Epoch [200/1000], Loss: 1.0242
    # Epoch [300/1000], Loss: 0.8017
    # Epoch [400/1000], Loss: 0.7163
    # Epoch [500/1000], Loss: 0.6836
    # Epoch [600/1000], Loss: 0.6710
    # Epoch [700/1000], Loss: 0.6662
    # Epoch [800/1000], Loss: 0.6643
    # Epoch [900/1000], Loss: 0.6636
    # Epoch [1000/1000], Loss: 0.6634
    # Learned weight: 1.9577, Learned bias: 3.2045
    # Predictions for [[4.0], [7.0]]: [[11.035286903381348], [16.90837860107422]]
    