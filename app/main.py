import streamlit as st
import torch
from src.models.semantic_search_model import SemanticSearchModel
from src.data_processing.data_loader import DataLoader
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Semantic Product Search",
    page_icon="🔍",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    model = SemanticSearchModel()
    model.load_state_dict(torch.load('semantic_search_model.pth'))
    model.eval()
    return model

@st.cache_data
def load_products():
    data_loader = DataLoader()
    df = data_loader.load_data()
    products_df = pd.read_parquet('data/shopping_queries_dataset_products.parquet')
    return products_df

def main():
    st.title("🔍 Semantic Product Search")
    st.write("Enter your search query to find relevant products using semantic search.")
    
    # Load model and data
    model = load_model()
    products_df = load_products()
    
    # Search interface
    query = st.text_input("Enter your search query:", placeholder="e.g., comfortable running shoes for women")
    
    if query:
        with st.spinner("Searching for relevant products..."):
            # Get product texts
            product_texts = []
            for _, row in products_df.iterrows():
                product_text = f"{row['product_title']} {row['product_description']}"
                product_texts.append(product_text)
            
            # Rank products
            ranked_products, scores = model.rank_products(query, product_texts, top_k=10)
            
            # Display results
            st.subheader("Search Results")
            
            for i, (product_text, score) in enumerate(zip(ranked_products, scores), 1):
                # Find the corresponding product in the dataframe
                product_row = products_df[
                    (products_df['product_title'] + " " + products_df['product_description']).str.contains(
                        product_text.split()[0],  # Use first word to find the product
                        case=False
                    )
                ].iloc[0]
                
                # Create expandable product card
                with st.expander(f"#{i} - {product_row['product_title']} (Score: {score:.4f})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Product Details:**")
                        st.write(f"**Title:** {product_row['product_title']}")
                        if pd.notna(product_row['product_brand']):
                            st.write(f"**Brand:** {product_row['product_brand']}")
                        if pd.notna(product_row['product_color']):
                            st.write(f"**Color:** {product_row['product_color']}")
                    
                    with col2:
                        st.write("**Description:**")
                        st.write(product_row['product_description'])
                        
                        if pd.notna(product_row['product_bullet_point']):
                            st.write("**Key Features:**")
                            st.write(product_row['product_bullet_point'])
            
            # Visualize score distribution
            st.subheader("Relevance Score Distribution")
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Distribution of Relevance Scores",
                labels={'x': 'Relevance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 