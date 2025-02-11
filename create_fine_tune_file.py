import pandas as pd
import json
from collections import Counter
from recommendation_model import RecommendationModel
from pathlib import Path

def create_recommendations_jsonl(input_csv_path, output_jsonl_path):
    #read csv data
    df = pd.read_csv(input_csv_path)
    
    #convert 'Order Date' to datetime for proper sorting
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Order Date'] = df['Order Date'].fillna(df['Order Date'].min())
    
    #sort rows by 'Order Date' ascending
    df_sorted = df.sort_values(by='Order Date', ascending=True)
    
    #initialize the RecommendationModel with the full dataset
    model = RecommendationModel(train_df=df_sorted)
    
    #set hybrid weights (adjust as needed)
    model.set_hybrid_weights(cf_weight=0.7, cb_weight=0.3)
    
    #group df_sorted by Customer_ID, build recommendation lines
    grouped = df_sorted.groupby('Customer_ID')
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_jsonl:
        for customer_id, group in grouped:
            #build "User Profile" info
            
            # a) city: pick the city for this user
            city_counts = group['City'].value_counts()
            if not city_counts.empty:
                main_city = city_counts.index[0]
            else:
                main_city = "Unknown"

            
            # b) bumber of total purchases
            total_purchases = len(group)
            
            # c) top 3 categories purchased
            category_counts = group['Product Category'].value_counts()
            top_categories = list(category_counts.index[:3])
            
            #create a short user profile text
            user_profile_text = f"User ID: {customer_id}, Main City: {main_city}, Total Purchases: {total_purchases}, "

            if top_categories:
                user_profile_text += f"Top Categories: {', '.join(top_categories)}"
            else:
                user_profile_text += "Top Categories: None"
            
            #Generate initial recommended products
            product_counts = group['Product'].value_counts()
            top_5_products = list(product_counts.index[:5])
            
            #determine how many more products are needed to reach 5
            num_current = len(top_5_products)
            num_to_fill = 5 - num_current
            
            if num_to_fill > 0:
                #fetch additional recommendations using the model
                additional_recommendations = model.get_hybrid_recommendations(
                    user_id=customer_id, 
                    top_n=num_to_fill
                )
                
                #convert Product IDs to Product Names 
                additional_product_names = []

                for pid in additional_recommendations:
                    product_name = model.cb_product_id_to_name.get(pid, "Unknown Product")
                    additional_product_names.append(product_name)
                                
                #append additional recommendations, no duplicates
                for prod in additional_product_names:
                    if prod not in top_5_products and len(top_5_products) < 5:
                        top_5_products.append(prod)
            
            #ensure exactly 5 recommendations
            recommended_5 = top_5_products[:5]
            
            #turn that list into a string
            recommended_5_str = ", ".join(recommended_5)
            
            #build the conversation structure
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"User Profile: {user_profile_text}\n\n"
                        "Recommend 5 products they might be interested in:"
                    )
                },
                {
                    "role": "assistant",
                    "content": recommended_5_str
                }
            ]
            
            #each line in the JSONL file is one JSON with the "messages" key
            json_line = {
                "messages": messages
            }
            
            #write to the JSONL file
            f_jsonl.write(json.dumps(json_line) + "\n")
    
    print(f"{output_jsonl_path} created successfully with the full dataset for fine-tuning!")

def get_recommended_products(user_id, top_n, model):
    #gets user id, return a list of recommended products using the hybrid recommendation model

    #generate top-n hybrid recommendations for the user
    recommended_product_ids = model.get_hybrid_recommendations(user_id=user_id, top_n=top_n)
    
    #retrieve product names using the model's mapping
    recommended_products = []

    for pid in recommended_product_ids:
        product_name = model.cb_product_id_to_name.get(pid, "Unknown Product")
        recommended_products.append(product_name)
    
    return recommended_products

if __name__ == "__main__":
    #get the directory of the current script
    SCRIPT_DIR = Path(__file__).resolve().parent

    #if data dir dont exists, create it
    data_dir = SCRIPT_DIR / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    #path to CSV relative path
    input_csv = data_dir / "train_set.csv"
    #path to JSONL relative path
    output_jsonl = data_dir / "gpt_finetune_data_train_set.jsonl"
    
    create_recommendations_jsonl(input_csv, output_jsonl)
