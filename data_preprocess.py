import pandas as pd
from collections import defaultdict
from pathlib import Path


#adding 3 columns - Customer_ID, Product_ID, behavior_rating

W_P = 5    #same product
W_SC = 3   #same sub-category
W_C = 1    #same category
W_Q = 0.2  #quantity factor


#get the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

#if data dir dont exists, create it
data_dir = SCRIPT_DIR / "data"
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

#path to CSV relative path
INPUT_CSV_PATH = data_dir / "Sales Data.csv"
OUTPUT_CSV_PATH = data_dir / "sales_data_final.csv"

df = pd.read_csv(INPUT_CSV_PATH)  

df['Customer_ID'] = df['Purchase Address'].astype('category').cat.codes + 1 #by unique address
df['Product_ID'] = df['Product'].astype('category').cat.codes + 1. # by unuqie catetgory and product name

df['Order Date'] = pd.to_datetime(df['Order Date'])
df.sort_values(by=['Customer_ID', 'Order Date'], inplace=True)

#new column, behavior_rating
df['behavior_rating'] = 0.0


#counters keyed by customer - product/sub cat/cat - number of purchases so far
cust_product_count = defaultdict(lambda: defaultdict(int))
cust_subcat_count  = defaultdict(lambda: defaultdict(int))
cust_cat_count     = defaultdict(lambda: defaultdict(int))

#iterate over rows in chronological order
for idx, row in df.iterrows():
    cust_id     = row['Customer_ID']
    product_id  = row['Product_ID']
    cat         = row['Product Category']
    quantity    = row['Quantity Ordered']
    
    #look up how many times they've purchased these in the past
    product_ct  = cust_product_count[cust_id][product_id]
    cat_ct      = cust_cat_count[cust_id][cat]
    
    #calculate behavior_rating
    rating = (W_P * product_ct + W_C * cat_ct + W_Q * quantity)
    
    #computed rating to the row
    df.at[idx, 'behavior_rating'] = rating
    
    #update the counters for the next iteration
    cust_product_count[cust_id][product_id] += 1
    cust_cat_count[cust_id][cat]            += 1

df.to_csv(OUTPUT_CSV_PATH, index=False)