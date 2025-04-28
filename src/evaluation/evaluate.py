import numpy as np
from typing import List, Tuple
import torch
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

def precision_at_k(y_true: List[int], y_pred: List[float], k: int) -> float:
    """Calculate Precision@K."""
    # Sort predictions and get top-k indices
    top_k_indices = np.argsort(y_pred)[-k:]
    
    # Count relevant items in top-k
    relevant_count = sum(1 for i in top_k_indices if y_true[i] >= 3)  # Consider E and S as relevant
    
    return relevant_count / k

def recall_at_k(y_true: List[int], y_pred: List[float], k: int) -> float:
    """Calculate Recall@K."""
    # Total number of relevant items
    total_relevant = sum(1 for score in y_true if score >= 3)
    
    if total_relevant == 0:
        return 0.0
    
    # Sort predictions and get top-k indices
    top_k_indices = np.argsort(y_pred)[-k:]
    
    # Count relevant items in top-k
    relevant_in_top_k = sum(1 for i in top_k_indices if y_true[i] >= 3)
    
    return relevant_in_top_k / total_relevant

def f1_at_k(y_true: List[int], y_pred: List[float], k: int) -> float:
    """Calculate F1@K."""
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def mean_average_precision(y_true: List[int], y_pred: List[float], k: int) -> float:
    """Calculate Mean Average Precision (MAP)@K."""
    # Sort predictions and get top-k indices
    top_k_indices = np.argsort(y_pred)[-k:]
    
    # Calculate average precision
    sum_precision = 0.0
    relevant_count = 0
    
    for i, idx in enumerate(top_k_indices, 1):
        if y_true[idx] >= 3:  # Consider E and S as relevant
            relevant_count += 1
            sum_precision += relevant_count / i
    
    if relevant_count == 0:
        return 0.0
    
    return sum_precision / relevant_count

def evaluate_model(
    model: torch.nn.Module,
    test_queries: List[str],
    test_products: List[str],
    test_labels: List[int],
    k_values: List[int] = [1, 3, 5, 10]
) -> dict:
    """Evaluate the model using various ranking metrics."""
    # Get predictions
    with torch.no_grad():
        similarity_scores = model(test_queries, test_products)
        predictions = similarity_scores.cpu().numpy()
    
    results = {}
    
    for k in k_values:
        # Calculate metrics
        precision = precision_at_k(test_labels, predictions, k)
        recall = recall_at_k(test_labels, predictions, k)
        f1 = f1_at_k(test_labels, predictions, k)
        map_score = mean_average_precision(test_labels, predictions, k)
        
        # Calculate NDCG
        ndcg = ndcg_score(
            np.array([test_labels]),
            np.array([predictions]),
            k=k
        )
        
        results[f'P@{k}'] = precision
        results[f'R@{k}'] = recall
        results[f'F1@{k}'] = f1
        results[f'MAP@{k}'] = map_score
        results[f'NDCG@{k}'] = ndcg
    
    return results

def plot_metrics(results: dict, k_values: List[int]):
    """Plot evaluation metrics."""
    metrics = ['P', 'R', 'F1', 'MAP', 'NDCG']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = [results[f'{metric}@{k}'] for k in k_values]
        plt.plot(k_values, values, marker='o')
        plt.title(f'{metric}@K')
        plt.xlabel('K')
        plt.ylabel('Score')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.close()

if __name__ == '__main__':
    from src.data_processing.data_loader import DataLoader
    from src.models.semantic_search_model import SemanticSearchModel
    
    # Load and preprocess data
    data_loader = DataLoader()
    df = data_loader.load_data()
    processed_df = data_loader.prepare_dataset(df)
    _, _, test_df = data_loader.split_data(processed_df)
    
    # Prepare test data
    test_queries = test_df['processed_query'].tolist()
    test_products = test_df['processed_product'].tolist()
    test_labels = test_df['relevance_score'].tolist()
    
    # Load trained model
    model = SemanticSearchModel()
    model.load_state_dict(torch.load('semantic_search_model.pth'))
    model.eval()
    
    # Evaluate model
    k_values = [1, 3, 5, 10]
    results = evaluate_model(model, test_queries, test_products, test_labels, k_values)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot metrics
    plot_metrics(results, k_values) 