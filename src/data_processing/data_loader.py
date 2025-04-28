import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DatasetLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """Load and combine the dataset files."""
        # Load examples
        examples_df = pd.read_parquet(self.data_dir / 'shopping_queries_dataset_examples.parquet')
        
        # Load products
        products_df = pd.read_parquet(self.data_dir / 'shopping_queries_dataset_products.parquet')
        
        # Load sources
        sources_df = pd.read_csv(self.data_dir / 'shopping_queries_dataset_sources.csv')
        
        # Merge datasets
        merged_df = examples_df.merge(
            products_df,
            on=['product_id', 'product_locale'],
            how='left'
        ).merge(
            sources_df,
            on='query_id',
            how='left'
        )
        
        return merged_df
    
    def preprocess_text(self, text):
        """Preprocess text by removing special characters, stopwords, and lemmatizing."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove stopwords and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def prepare_product_text(self, row):
        """Combine and preprocess product title and description."""
        title = self.preprocess_text(row['product_title'])
        description = self.preprocess_text(row['product_description'])
        
        # Combine title and description with a separator
        return f"{title} {description}"
    
    def prepare_dataset(self, df):
        """Prepare the dataset for training by combining and preprocessing text fields."""
        # Preprocess queries
        df['processed_query'] = df['query'].apply(self.preprocess_text)
        
        # Combine and preprocess product information
        df['processed_product'] = df.apply(self.prepare_product_text, axis=1)
        
        # Create relevance labels (convert ESCI labels to numerical scores)
        label_mapping = {
            'E': 4,  # Exact
            'S': 3,  # Substitute
            'C': 2,  # Complement
            'I': 1   # Irrelevant
        }
        df['relevance_score'] = df['esci_label'].map(label_mapping)
        
        return df[['processed_query', 'processed_product', 'relevance_score', 'split']]
    
    def split_data(self, df):
        """Split the dataset into train, validation, and test sets."""
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        return train_df, val_df, test_df 