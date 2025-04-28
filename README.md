# Semantic Product Search

This project implements a deep learning-based semantic product search system using transformer models. The system accepts natural language queries and returns ranked product results based on semantic relevance.

## Project Structure
```
semantic-product-search/
├── data/                    # Directory for dataset files
├── src/                     # Source code directory
│   ├── data_processing/     # Data preprocessing scripts
│   ├── models/             # Model implementation
│   ├── training/           # Training scripts
│   └── evaluation/         # Evaluation scripts
├── notebooks/              # Jupyter notebooks for exploration
├── app/                    # Streamlit web application
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/amsepi/semantic-product-search.git
cd semantic-product-search
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Visit https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset
- Download the following files:
  - shopping_queries_dataset_examples.parquet
  - shopping_queries_dataset_products.parquet
  - shopping_queries_dataset_sources.csv
- Place them in the `data/` directory

## Project Components

1. **Data Processing**
   - Text preprocessing (lowercase, stop words removal, lemmatization)
   - Data splitting (70-15-15 for train-val-test)
   - Feature engineering

2. **Model Development**
   - Implementation of transformer-based models
   - Training and validation
   - Hyperparameter optimization

3. **Evaluation**
   - NDCG, MAP, Precision@K, Recall@K, F1@K metrics
   - Performance visualization

4. **Web Application**
   - Streamlit-based interface
   - Real-time search functionality
   - Results visualization

## Running the Application

To start the Streamlit application:
```bash
streamlit run app/main.py
```

## License
MIT License 