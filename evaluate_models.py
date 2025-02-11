
import os
import pandas as pd
from recommendation_model import RecommendationModel
import openai
from dotenv import load_dotenv
from tqdm import tqdm  # progress bars
import time
from pathlib import Path

def load_product_mapping(products_csv_path):# load the product mapping from Product_ID to Product name
    try:
        products_df = pd.read_csv(products_csv_path)
        product_mapping = pd.Series(products_df.Product.values, index=products_df.Product_ID).to_dict()
        return product_mapping
    except Exception as e:
        raise Exception(f"Error loading products CSV file: {e}")


def main():
    #get the directory of the current script
    SCRIPT_DIR = Path(__file__).resolve().parent

    #if data dir dont exists, create it
    data_dir = SCRIPT_DIR / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    #if models dir dont exists, create it
    models_dir = SCRIPT_DIR / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)


    #load environment variables
    load_dotenv(SCRIPT_DIR / '.env')
    
    #define paths to models and data
    CF_MODEL_PATH = models_dir / "collaborative_model.pkl"
    CB_MODELS_DIR = models_dir
    TRAIN_SET_PATH = data_dir / "train_set.csv"
    TEST_SET_PATH = data_dir / "test_set.csv"
    PRODUCTS_CSV_PATH = data_dir / "products.csv"
    
    #load OpenAI API key and fine-tuned model name from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    fine_tuned_model = os.getenv("FINE_TUNED_MODEL")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    if not fine_tuned_model:
        raise ValueError("Fine-tuned model name not found. Please set the FINE_TUNED_MODEL environment variable.")
    
    #check if the required files exist
    required_files = [CF_MODEL_PATH, TRAIN_SET_PATH, TEST_SET_PATH, PRODUCTS_CSV_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found at {file_path}")
    if not os.path.isdir(CB_MODELS_DIR):
        raise FileNotFoundError(f"Content-Based models directory not found at {CB_MODELS_DIR}")
    
    #load product mapping
    print("Loading product mapping...")
    product_mapping = load_product_mapping(PRODUCTS_CSV_PATH)
    
    #load training and test datasets
    print("Loading training and test datasets...")
    train_df = pd.read_csv(TRAIN_SET_PATH)
    test_df = pd.read_csv(TEST_SET_PATH)
    
    # initialize the RecommendationModel with loaded=True to skip training
    print("Initializing RecommendationModel...")
    model = RecommendationModel(loaded=True)
    
    #assign the loaded dataframes to the model
    model.train_df = train_df.copy()
    model.test_df = test_df.copy().tail(20)  # Using the last entries for testing
    
    #manually prepare the Collaborative Filtering test set
    if 'behavior_rating' not in model.test_df.columns:
        raise ValueError("The test set must contain the 'behavior_rating' column.")
    model.cf_testset = list(zip(
        model.test_df['Customer_ID'],
        model.test_df['Product_ID'],
        model.test_df['behavior_rating']
    ))
    
    #lad the pre-trained Collaborative Filtering and Content-Based models
    print("Loading Collaborative Filtering and Content-Based models...")
    model.load_all_models(cf_path=CF_MODEL_PATH, cb_directory=CB_MODELS_DIR)
    
    #adjust the weights
    # model.set_hybrid_weights(cf_weight=0.7, cb_weight=0.3)
    
    
    print("\nEvaluating Collaborative Filtering Hit Rate...")
    model.evaluate_collaborative_metrics(top_n=5)
    
    print("\nEvaluating Content-Based Recommendation Hit Rate...")
    model.evaluate_content_metrics(top_n=5)
    
    print("\nEvaluating Hybrid Recommendation Hit Rate...")
    model.evaluate_hybrid_metrics(top_n=5)
    
    # ealuate GPT Fine Tuned Model Hit Rate
    gpt_hit_rate = model.evaluate_gpt_model(
        test_df=model.test_df,
        product_mapping=product_mapping,
        fine_tuned_model=fine_tuned_model,
        openai_api_key=openai_api_key,
        top_n=5
    )
    
    

if __name__ == "__main__":
    main()
