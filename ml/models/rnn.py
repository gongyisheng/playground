# # Problem: Implement an RNN in PyTorch

# ### Problem Statement
# You are tasked with implementing a **Recurrent Neural Network (RNN)** in PyTorch to process sequential data. The model should contain an RNN layer for handling sequential input and a fully connected layer to output the final predictions. Your goal is to complete the RNN model by defining the necessary layers and implementing the forward pass.

# ### Requirements
# 1. **Define the RNN Model**:
#    - Add an **RNN layer** to process sequential data.
#    - Add a **fully connected layer** to map the RNN output to the final prediction.

# ### Constraints
# - Use appropriate configurations for the RNN layer, including hidden units and input/output sizes.


# <details>
#   <summary>💡 Hint</summary>
#   Add the RNN layer (self.rnn) and fully connected layer (self.fc) in RNNModel.__init__.
#   <br>
#   Implement the forward pass to process inputs through the RNN layer and fully connected layer.
# </details>

import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic sequential data
def generate_data():
    torch.manual_seed(42)
    sequence_length = 10
    num_samples = 100

    # Create a sine wave dataset
    X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
    Y = torch.sin(X)

    # Prepare data for RNN
    in_seq = []
    out_seq = []
    for i in range(len(Y) - sequence_length):
        in_seq.append(Y[i:i + sequence_length])
        out_seq.append(Y[i + sequence_length])
    return torch.stack(in_seq), torch.stack(out_seq)

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Weight matrices for input and hidden state
        self.W_ih = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.tanh(x_t @ self.W_ih + h_t @ self.W_hh + self.b_h)

        output = self.output_layer(h_t)
        return output

def train(model, epochs, lr):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_seq, Y_seq = generate_data()
    model.train()

    # Training loop
    for epoch in range(epochs):
        for sequences, labels in zip(X_seq, Y_seq):
            sequences = sequences.unsqueeze(0)  # Add batch dimension
            labels = labels.unsqueeze(0)  # Add batch dimension

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

def eval(model):
    model.eval()
    # Testing on new data
    X_test = torch.linspace(4 * 3.14159, 5 * 3.14159, steps=10).unsqueeze(1)
    # Reshape to (batch_size, sequence_length, input_size)
    X_test = X_test.unsqueeze(0)  # Add batch dimension, shape becomes (1, 10, 1)

    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for new sequence: {predictions.tolist()}")

if __name__ == "__main__":
    model = RNNModel()

    train(model, 100, 0.001)
    eval(model)
    
