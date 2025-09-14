import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ml.transformer.transformer import Transformer
import numpy as np

# Example dataset for translation
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        
        # Convert words to indices
        src_indices = [self.src_vocab.get(word, self.src_vocab['<UNK>']) for word in src]
        tgt_indices = [self.tgt_vocab.get(word, self.tgt_vocab['<UNK>']) for word in tgt]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape output and target for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Reshape output and target for loss calculation
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example vocabulary
    src_vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3, 'the': 4, 'cat': 5, 'sat': 6, 'on': 7, 'mat': 8}
    tgt_vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3, 'le': 4, 'chat': 5, 's\'est': 6, 'assis': 7, 'sur': 8, 'tapis': 9}
    
    # Example data
    src_sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'cat', 'sat', 'on', 'the', 'mat']
    ]
    tgt_sentences = [
        ['le', 'chat', 's\'est', 'assis', 'sur', 'le', 'tapis'],
        ['le', 'chat', 's\'est', 'assis', 'sur', 'le', 'tapis']
    ]
    
    # Create dataset and dataloader
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        # Generate example translation
        src = torch.tensor([[src_vocab[word] for word in src_sentences[0]]]).to(device)
        generated = model.generate(src)
        print(f'Generated translation: {generated[0].tolist()}')

if __name__ == '__main__':
    main() 