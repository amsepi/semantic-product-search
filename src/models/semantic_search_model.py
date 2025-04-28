import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple
import numpy as np

class SemanticSearchModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super(SemanticSearchModel, self).__init__()
        
        # Initialize BERT model and tokenizer
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Projection layer to map BERT embeddings to a common space
        self.projection = nn.Linear(768, 256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text input using BERT."""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**inputs)
            
        # Use [CLS] token embedding as the sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to common space
        projected = self.projection(cls_embedding)
        projected = self.dropout(projected)
        
        return projected.squeeze(0)  # Remove batch dimension
    
    def compute_similarity(self, query_embedding: torch.Tensor, 
                         product_embedding: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and product embeddings."""
        # Ensure both tensors are 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if product_embedding.dim() == 1:
            product_embedding = product_embedding.unsqueeze(0)
            
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        product_norm = torch.nn.functional.normalize(product_embedding, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(query_norm, product_norm.t())
        
        return similarity
    
    def forward(self, queries: List[str], products: List[str]) -> torch.Tensor:
        """Forward pass to compute similarity scores between queries and products."""
        # Encode queries and products
        query_embeddings = []
        product_embeddings = []
        
        for q in queries:
            query_embeddings.append(self.encode_text(q))
        for p in products:
            product_embeddings.append(self.encode_text(p))
            
        # Stack embeddings
        query_embeddings = torch.stack(query_embeddings)
        product_embeddings = torch.stack(product_embeddings)
        
        # Compute similarity scores
        similarity_scores = self.compute_similarity(query_embeddings, product_embeddings)
        
        return similarity_scores
    
    def rank_products(self, query: str, products: List[str], 
                     top_k: int = 10) -> Tuple[List[str], List[float]]:
        """Rank products based on their similarity to the query."""
        # Encode query and products
        query_embedding = self.encode_text(query)
        product_embeddings = torch.stack([self.encode_text(p) for p in products])
        
        # Compute similarity scores
        similarity_scores = self.compute_similarity(query_embedding, product_embeddings)
        
        # Get top-k products and their scores
        scores, indices = torch.topk(similarity_scores.squeeze(), k=min(top_k, len(products)))
        
        # Convert to lists
        ranked_products = [products[i] for i in indices]
        ranked_scores = scores.tolist()
        
        return ranked_products, ranked_scores 