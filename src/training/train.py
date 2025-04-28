import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

class ProductSearchDataset(Dataset):
    def __init__(self, queries: List[str], products: List[str], labels: List[int]):
        self.queries = queries
        self.products = products
        self.labels = labels
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return {
            'query': self.queries[idx],
            'product': self.products[idx],
            'label': self.labels[idx]
        }

def train_model(
    model: nn.Module,
    train_data: Tuple[List[str], List[str], List[int]],
    val_data: Tuple[List[str], List[str], List[int]],
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the semantic search model."""
    # Move model to device
    model = model.to(device)
    
    # Create datasets
    train_dataset = ProductSearchDataset(*train_data)
    val_dataset = ProductSearchDataset(*val_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            queries = batch['query']
            products = batch['product']
            labels = batch['label'].float().to(device)
            
            # Forward pass
            similarity_scores = model(queries, products)
            loss = criterion(similarity_scores, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                queries = batch['query']
                products = batch['product']
                labels = batch['label'].float().to(device)
                
                similarity_scores = model(queries, products)
                loss = criterion(similarity_scores, labels)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # Save the plot
    plt.savefig('training_curves.png')
    plt.close()
    
    # Save the model
    torch.save(model.state_dict(), 'semantic_search_model.pth')
    
    return model, train_losses, val_losses

if __name__ == '__main__':
    from src.data_processing.data_loader import DatasetLoader
    from src.models.semantic_search_model import SemanticSearchModel
    
    # Load and preprocess data
    data_loader = DatasetLoader()
    df = data_loader.load_data()
    processed_df = data_loader.prepare_dataset(df)
    train_df, val_df, test_df = data_loader.split_data(processed_df)
    
    # Prepare training data
    train_data = (
        train_df['processed_query'].tolist(),
        train_df['processed_product'].tolist(),
        train_df['relevance_score'].tolist()
    )
    
    # Prepare validation data
    val_data = (
        val_df['processed_query'].tolist(),
        val_df['processed_product'].tolist(),
        val_df['relevance_score'].tolist()
    )
    
    # Initialize and train model
    model = SemanticSearchModel()
    trained_model, train_losses, val_losses = train_model(
        model,
        train_data,
        val_data,
        batch_size=32,
        num_epochs=10
    ) 