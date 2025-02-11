import os
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from surprise import Dataset, Reader, SVD
from surprise.accuracy import rmse, mae
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
from tqdm import tqdm  #progress bars
import time 
from pathlib import Path

class RecommendationModel:
    def __init__(self, train_df=None, test_df=None, loaded=False):
        #initialize hybrid weights
        self.cf_weight = 0.5
        self.cb_weight = 0.5

        if not loaded and train_df is not None:
            self.train_df = train_df.copy()
            if test_df is not None:
                self.test_df = test_df.copy()
            else:
                self.test_df = None


            #convert 'order date' to datetime
            self.train_df['Order Date'] = pd.to_datetime(
                self.train_df['Order Date'], errors='coerce'
            )
            self.train_df['Order Date'] = self.train_df['Order Date'].fillna(
                self.train_df['Order Date'].min()
            )

            if self.test_df is not None:
                self.test_df['Order Date'] = pd.to_datetime(
                    self.test_df['Order Date'], errors='coerce'
                )
                self.test_df['Order Date'] = self.test_df['Order Date'].fillna(
                    self.test_df['Order Date'].min()
                )

            #initialize placeholders for collaborative filtering
            self.cf_algo = None
            self.cf_testset = None

            #initialize placeholders for content-based
            self.cb_tfidf_vectorizer = None
            self.cb_content_similarity = None
            self.cb_product_id_to_idx = {}
            self.cb_idx_to_product_id = {}
            self.cb_all_products = None
            self.cb_product_id_to_name = {}

            #build both models
            self.build_collaborative_model()
            self.build_content_model()
        else:
            #when loaded=true, models will be loaded separately
            self.train_df = None
            self.test_df = None

            #collaborative filtering placeholders
            self.cf_algo = None
            self.cf_testset = None

            #content-based placeholders
            self.cb_tfidf_vectorizer = None
            self.cb_content_similarity = None
            self.cb_product_id_to_idx = {}
            self.cb_idx_to_product_id = {}
            self.cb_all_products = None
            self.cb_product_id_to_name = {}

    # collaborative filtering methods
    def build_collaborative_model(self):#build and train the collaborative filtering model using svd

        if self.train_df is None:
            raise ValueError("training dataframe not loaded. cannot build collaborative model.")

        #ensure 'behavior_rating' column exists
        if 'behavior_rating' not in self.train_df.columns:
            raise ValueError("missing required column: 'behavior_rating'")

        #build a surprise trainset from train_df
        reader = Reader(rating_scale=(self.train_df['behavior_rating'].min(),
                                      self.train_df['behavior_rating'].max()))
        train_data = Dataset.load_from_df(
            self.train_df[['Customer_ID', 'Product_ID', 'behavior_rating']],
            reader
        )
        trainset = train_data.build_full_trainset()

        #train the svd model
        algo = SVD(random_state=42)
        algo.fit(trainset)
        self.cf_algo = algo

        #prepare the testset if provided
        if self.test_df is not None:
            #ensure 'behavior_rating' column exists in test_df
            if 'behavior_rating' not in self.test_df.columns:
                raise ValueError("missing required column: 'behavior_rating' in test dataframe.")

            #create the testset for evaluation
            self.cf_testset = list(zip(
                self.test_df['Customer_ID'],
                self.test_df['Product_ID'],
                self.test_df['behavior_rating']
            ))

    def get_svd_predictions(self, user_id, user_products, top_n=5):#get top n predictions from svd for a user
        if self.cf_algo is None or self.train_df is None:
            raise ValueError("collaborative filtering model not loaded or training dataframe not available.")

        #get all unique products from training data
        all_products = self.train_df['Product_ID'].unique()

        #filter products not interacted with by the user
        products_to_predict = []

        for product in all_products:
            if product not in user_products:
                products_to_predict.append(product)

        #get SVD predictions for each candidate product
        predictions = []

        for pid in products_to_predict:
            predictions.append(self.cf_algo.predict(user_id, pid))


        #sort by estimated rating in descending order
        predictions.sort(key=lambda x: x.est, reverse=True)

        #return top_n predictions
        return predictions[:top_n]

    def evaluate_collaborative_metrics(self, top_n=5):#evaluate collaborative filtering model using precision, recall, f1, and hit rate
        if self.test_df is None:
            raise ValueError("test dataframe not loaded. cannot evaluate collaborative metrics.")

        hits = 0
        total = 0

        test_grouped = self.test_df.groupby('Customer_ID')

        for user_id, group in test_grouped:
            actual_product_ids = group['Product_ID'].unique()
            user_products = self.train_df[self.train_df['Customer_ID'] == user_id]['Product_ID'].unique()

            predicted_predictions = self.get_svd_predictions(user_id=user_id, user_products=user_products, top_n=top_n)
            predicted_product_ids = []
            for pred in predicted_predictions:
                predicted_product_ids.append(pred.iid)


            for actual_product_id in actual_product_ids:
                if actual_product_id in predicted_product_ids:
                    hits += 1
            total += len(actual_product_ids)

        if total == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            recall = hits / total
            precision = hits / (top_n * total)
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0


        print(f"Collaborative Filtering Metrics@N={top_n}:")
        print(f"precision: {precision:.2%}")
        print(f"recall: {recall:.2%}")
        print(f"f1 score: {f1:.2%}")
        print(f"hits: {hits}/{total}")

        return precision, recall, f1

    def evaluate_collaborative_hit_rate(self, top_n=5):# evaluate the collaborative filtering model hit rate
        if self.test_df is None:
            raise ValueError("test dataframe not loaded. cannot evaluate collaborative hit rate.")

        #initialize counters
        hits = 0
        total = 0

        #group test interactions by user for efficiency
        test_grouped = self.test_df.groupby('Customer_ID')

        for user_id, group in test_grouped:
            actual_product_ids = group['Product_ID'].unique()
            user_products = self.train_df[self.train_df['Customer_ID'] == user_id]['Product_ID'].unique()

            #generate top-n svd predictions excluding already interacted products
            predicted_predictions = self.get_svd_predictions(user_id=user_id, user_products=user_products, top_n=top_n)
            predicted_product_ids = []

            for pred in predicted_predictions:
                predicted_product_ids.append(pred.iid)


            #check if any actual product is among the top-n predictions
            for actual_product_id in actual_product_ids:
                if actual_product_id in predicted_product_ids:
                    #print(f"cf - actual: {actual_product_id}, recommended: {predicted_product_ids}")
                    hits += 1
            total += len(actual_product_ids)

        hit_rate = hits / total if total else 0.0
        print(f"Collaborative Filtering Hit Rate: {hit_rate:.2%} ({hits}/{total})")
        return hit_rate

    def save_collaborative_model(self, file_path):#save the collaborative filtering model to the specified file path.
        with open(file_path, 'wb') as f:
            pickle.dump(self.cf_algo, f)
        print(f"collaborative filtering model saved to {file_path}.")

    def load_collaborative_model(self, file_path):#load the collaborative filtering model from the specified file path.
        with open(file_path, 'rb') as f:
            self.cf_algo = pickle.load(f)
        print(f"collaborative filtering model loaded from {file_path}.")

    
    # content-based recommendation methods
    def prepare_content_data(self):#prepares the dataset by ensuring necessary columns are present and handling missing values.
        if self.train_df is None:
            raise ValueError("training dataframe not loaded. initialize with train_df or set loaded=true and load models.")

        #define required columns based on user request
        required_columns = [
            'Customer_ID', 'Product_ID', 'City', 'Product', 'Product Category', 'Price Each', 'Order Date'
        ]
        for col in required_columns:
            if col not in self.train_df.columns:
                raise ValueError(f"missing required column: {col}")

        #keep track of all products in the dataset
        self.cb_all_products = self.train_df['Product_ID'].unique()

    def build_content_model(self):# creates a tf-idf-based similarity matrix using specified categorical features
        if self.train_df is None:
            raise ValueError("training dataframe not loaded. cannot build content-based model.")

        self.prepare_content_data()

        #extract unique product information
        product_info = self.train_df.drop_duplicates('Product_ID')

        #fill missing values with empty strings (except for 'price each' which is numeric)
        for col in ['City', 'Product', 'Product Category']:
            product_info[col] = product_info[col].fillna('')

        #handle 'price each' by converting it to string and filling missing values
        product_info['Price Each'] = product_info['Price Each'].fillna(0).astype(str)

        #combine categorical fields into a single text feature for tf-idf
        product_info['combined_text'] = (
            product_info['City'].astype(str) + ' ' +
            product_info['Product'].astype(str) + ' ' +
            product_info['Product Category'].astype(str) + ' ' +
            product_info['Price Each'].astype(str)
        )

        #create mappings between product_id and indices
        self.cb_product_id_to_idx = {}
        self.cb_idx_to_product_id = {}

        for idx, pid in enumerate(product_info['Product_ID']):
            self.cb_product_id_to_idx[pid] = idx
            self.cb_idx_to_product_id[idx] = pid


        #initialize and fit tf-idf vectorizer
        self.cb_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.cb_tfidf_vectorizer.fit_transform(product_info['combined_text'])

        #compute cosine similarity matrix
        self.cb_content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        #create product_id -> product_name mapping
        if 'Product' in self.train_df.columns:
            product_names = product_info[['Product_ID', 'Product']]
            self.cb_product_id_to_name = pd.Series(
                product_names['Product'].values,
                index=product_names['Product_ID']
            ).to_dict()
        else:
            # Initialize a dictionary mapping Product_ID to "unknown product"
            self.cb_product_id_to_name = {}

            for pid in product_info['Product_ID']:
                self.cb_product_id_to_name[pid] = "unknown product"


    def get_content_based_scores(self, user_id, top_n=5):#get top n content-based scores for a user
        if self.train_df is None or self.cb_content_similarity is None:
            raise ValueError("content-based model not loaded or dataframe not available.")

        #retrieve user purchased products
        user_data = self.train_df[self.train_df['Customer_ID'] == user_id]
        purchased_ids = user_data['Product_ID'].unique()


        #map purchased items to indices
        purchased_indices = []

        for pid in purchased_ids:
            if pid in self.cb_product_id_to_idx:
                purchased_indices.append(self.cb_product_id_to_idx[pid])


        #aggregate similarity scores for all products based on purchased items
        product_scores = {}
        for pid, idx in self.cb_product_id_to_idx.items():

            sim_values = self.cb_content_similarity[idx, purchased_indices]
            if len(sim_values) > 0:
                product_scores[pid] = sim_values.mean()
            else:
                product_scores[pid] = 0.0


        #sort products by similarity score in descending order and select top_n
        sorted_by_sim = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        top_n_content = dict(sorted_by_sim[:top_n])
        return top_n_content

    def get_top_n_recommendations(self, user_id, n=5):#top n recommendations based  on content-based filtering
        content_scores = self.get_content_based_scores(user_id, top_n=n)
        return list(content_scores.keys())

    def evaluate_content_metrics(self, top_n=5):#evaluate content-based model using precision, recall, f1, and hit rate
        if self.test_df is None:
            raise ValueError("test dataframe not loaded. cannot evaluate content metrics.")

        hits = 0
        total = 0

        for _, row in self.test_df.iterrows():
            user_id = row['Customer_ID']
            actual_product_id = row['Product_ID']

            recommended_products = self.get_top_n_recommendations(user_id, n=top_n)

            if len(recommended_products) == 0:
                continue

            total += 1
            if actual_product_id in recommended_products:
                hits += 1

        if total == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            recall = hits / total
            precision = hits / (top_n * total)
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0


        print(f"Content-Based Metrics@N={top_n}:")
        print(f"precision: {precision:.2%}")
        print(f"recall: {recall:.2%}")
        print(f"f1 score: {f1:.2%}")
        print(f"hits: {hits}/{total}")

        return precision, recall, f1

    def save_content_models(self, directory_path):#saves the content-based model components and necessary mappings to the specified directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        #save tf-idf vectorizer
        with open(os.path.join(directory_path, 'cb_tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.cb_tfidf_vectorizer, f)

        #save content similarity matrix (as joblib)
        joblib.dump(self.cb_content_similarity, os.path.join(directory_path, 'cb_content_similarity.joblib'))

        #save product index mappings
        with open(os.path.join(directory_path, 'cb_product_id_to_idx.pkl'), 'wb') as f:
            pickle.dump(self.cb_product_id_to_idx, f)
        with open(os.path.join(directory_path, 'cb_idx_to_product_id.pkl'), 'wb') as f:
            pickle.dump(self.cb_idx_to_product_id, f)

        #save product id -> name mapping
        with open(os.path.join(directory_path, 'cb_product_id_to_name.pkl'), 'wb') as f:
            pickle.dump(self.cb_product_id_to_name, f)

        #save all_products
        with open(os.path.join(directory_path, 'cb_all_products.pkl'), 'wb') as f:
            pickle.dump(self.cb_all_products, f)

        #save hybrid weights
        with open(os.path.join(directory_path, 'hybrid_weights.pkl'), 'wb') as f:
            pickle.dump({'cf_weight': self.cf_weight, 'cb_weight': self.cb_weight}, f)

        print(f"content-based models and components have been saved to {directory_path}.")

    def load_content_models(self, directory_path):#loads the content-based model components and necessary mappings from the specified directory
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"the directory {directory_path} does not exist.")

        with open(os.path.join(directory_path, 'cb_tfidf_vectorizer.pkl'), 'rb') as f:
            self.cb_tfidf_vectorizer = pickle.load(f)

        self.cb_content_similarity = joblib.load(
            os.path.join(directory_path, 'cb_content_similarity.joblib')
        )
        with open(os.path.join(directory_path, 'cb_product_id_to_idx.pkl'), 'rb') as f:
            self.cb_product_id_to_idx = pickle.load(f)
        with open(os.path.join(directory_path, 'cb_idx_to_product_id.pkl'), 'rb') as f:
            self.cb_idx_to_product_id = pickle.load(f)
        with open(os.path.join(directory_path, 'cb_product_id_to_name.pkl'), 'rb') as f:
            self.cb_product_id_to_name = pickle.load(f)
        with open(os.path.join(directory_path, 'cb_all_products.pkl'), 'rb') as f:
            self.cb_all_products = pickle.load(f)

        # load hybrid weights
        hybrid_weights_path = os.path.join(directory_path, 'hybrid_weights.pkl')
        if os.path.exists(hybrid_weights_path):
            with open(hybrid_weights_path, 'rb') as f:
                weights = pickle.load(f)
                self.cf_weight = weights.get('cf_weight', 0.5)
                self.cb_weight = weights.get('cb_weight', 0.5)

        print(f"content-based models and components have been loaded from {directory_path}.")

    # hybrid recommendation methods
    def set_hybrid_weights(self, cf_weight=0.5, cb_weight=0.5):
        #set the weights for collaborative filtering and content-based components in the hybrid model

        total = cf_weight + cb_weight
        if total == 0:
            raise ValueError("the sum of cf_weight and cb_weight must be greater than 0.")
        self.cf_weight = cf_weight / total
        self.cb_weight = cb_weight / total
        print(f"hybrid weights set to cf: {self.cf_weight}, cb: {self.cb_weight}")

    def get_hybrid_recommendations(self, user_id, top_n=5):#get top n hybrid recommendations for a user by combining cf and cb scores
        if self.cf_algo is None or self.cb_content_similarity is None:
            raise ValueError("both collaborative filtering and content-based models must be loaded.")

        #get users interacted products
        user_data = self.train_df[self.train_df['Customer_ID'] == user_id]
        user_products = user_data['Product_ID'].unique()

        #get cf predictions
        cf_predictions = self.get_svd_predictions(user_id=user_id, user_products=user_products, top_n=None)
        cf_scores = {}
        for pred in cf_predictions:
            cf_scores[pred.iid] = pred.est


        # get cb scores
        cb_scores = self.get_content_based_scores(user_id=user_id, top_n=None)

        #combine scores
        combined_scores = {}

        #normalize cf scores
        normalized_cf = {}

        if cf_scores:
            cf_max = max(cf_scores.values())
            cf_min = min(cf_scores.values())

            if cf_max != cf_min:
                normalized_cf = {}
                for pid, score in cf_scores.items():
                    normalized_cf[pid] = (score - cf_min) / (cf_max - cf_min)
            else:
                for pid in cf_scores:
                    normalized_cf[pid] = 1.0


        #normalize cb scores
        normalized_cb = {}

        if cb_scores:
            cb_max = max(cb_scores.values())
            cb_min = min(cb_scores.values())

            if cb_max != cb_min:
                normalized_cb = {}
                for pid, score in cb_scores.items():
                    normalized_cb[pid] = (score - cb_min) / (cb_max - cb_min)
            else:
                for pid in cb_scores:
                    normalized_cb[pid] = 1.0


        #combine normalized scores using weights
        all_candidate_pids = set(normalized_cf.keys()).union(set(normalized_cb.keys()))
        for pid in all_candidate_pids:
            cf_score = normalized_cf.get(pid, 0)
            cb_score = normalized_cb.get(pid, 0)
            combined_score = self.cf_weight * cf_score + self.cb_weight * cb_score
            combined_scores[pid] = combined_score

        #sort products by combined score in descending order
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        #elect top-n recommendations, excluding already interacted products
        top_recommendations = []
        for pid, score in sorted_combined:
            if pid not in user_products:
                top_recommendations.append(pid)
            if len(top_recommendations) == top_n:
                break

        return top_recommendations

    def evaluate_hybrid_metrics(self, top_n=5):#evaluate hybrid model using precision, recall, f1, and hit rate
        if self.test_df is None:
            raise ValueError("test dataframe not loaded. cannot evaluate hybrid metrics.")

        hits = 0
        total = 0

        for _, row in self.test_df.iterrows():
            user_id = row['Customer_ID']
            actual_product_id = row['Product_ID']

            recommended_products = self.get_hybrid_recommendations(user_id, top_n=top_n)

            if len(recommended_products) == 0:
                continue

            total += 1
            if actual_product_id in recommended_products:
                hits += 1

        if total == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            recall = hits / total
            precision = hits / (top_n * total)
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0


        print(f"Hybrid Model Metrics@N={top_n}:")
        print(f"precision: {precision:.2%}")
        print(f"recall: {recall:.2%}")
        print(f"f1 score: {f1:.2%}")
        print(f"hits: {hits}/{total}")

        return precision, recall, f1

    # combined save and load methods
    def save_all_models(self, cf_path='cf_model.pkl', cb_directory='cb_model_directory'):
        self.save_collaborative_model(cf_path)
        self.save_content_models(cb_directory)
        print("all models have been saved.")

    def load_all_models(self, cf_path='cf_model.pkl', cb_directory='cb_model_directory'):
        self.load_collaborative_model(cf_path)
        self.load_content_models(cb_directory)
        print("all models have been loaded.")


    def construct_user_profile(self, row):#construct a user profile
        customer_id = row['Customer_ID']
        
        # a) city: since it's a single row, use the city from the row
        if pd.notnull(row['City']):
            main_city = row['City']
        else:
            main_city = "unknown"

        
        # b) number of total purchases
        total_purchases = row.get('Total Purchases', 0)
        
        # c) top 3 categories purchased
        top_categories = row.get('Top Categories', "")
        if isinstance(top_categories, str):
            top_categories_list = []

            for cat in top_categories.split(','):
                top_categories_list.append(cat.strip())

            top_categories = top_categories_list

        else:
            top_categories = []
        top_categories = top_categories[:3]
        
        # create user profile text
        user_profile_text = f"user id: {customer_id}, main city: {main_city}, total purchases: {total_purchases}, "

        if top_categories:
            user_profile_text += f"top categories: {', '.join(top_categories)}"
        else:
            user_profile_text += "top categories: none"
        
        return user_profile_text

    def get_recommendations(self, user_profile, model_name, api_key, max_retries=3, wait_seconds=2):
        #get product recommendations from the fine-tuned model
        openai.api_key = api_key
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"user profile: {user_profile}\n\n"
                    "recommend 5 products they might be interested in:"
                )
            }
        ]
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7,
                    n=1,
                    stop=None
                )
                assistant_content = response.choices[0].message['content'].strip()
                return self.parse_recommendations(assistant_content)
            except openai.error.OpenAIError as e:
                print(f"openai api error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print(f"retrying in {wait_seconds} seconds...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print("max retries reached. skipping this user.")
                    return []
        return []

    def parse_recommendations(self, assistant_response):#parse the assistant response to extract recommended products.

        #split the response by commas or newlines and strip whitespace
        recommended_products = []

        if ',' in assistant_response:
            split_response = assistant_response.split(',')
        elif '\n' in assistant_response:
            split_response = assistant_response.split('\n')
        else:
            #split by spaces
            split_response = assistant_response.split()

        #clean and ensure exactly 5 recommendations
        recommended_products = []
        for prod in split_response:
            cleaned_prod = prod.strip()
            if cleaned_prod:
                recommended_products.append(cleaned_prod)


                return recommended_products[:5]


    def evaluate_gpt_model(self, test_df, product_mapping, fine_tuned_model, openai_api_key, top_n=5):
        #evaluate the gpt model using precision, recall, f1, and hit rate
        hits = 0
        total = 0

        print("\nevaluating gpt fine-tuned model metrics...")
        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
            user_profile = self.construct_user_profile(row)
            recommended_products = self.get_recommendations(user_profile, fine_tuned_model, openai_api_key)
            actual_product_id = row['Product_ID']
            actual_product = product_mapping.get(actual_product_id, None)

            if actual_product is None:
                continue

            total += 1
            if actual_product in recommended_products:
                hits += 1

        if total == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            recall = hits / total
            precision = hits / (top_n * total)
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0


        print(f"gpt model metrics@N={top_n}:")
        print(f"precision: {precision:.2%}")
        print(f"recall: {recall:.2%}")
        print(f"f1 score: {f1:.2%}")
        print(f"hits: {hits}/{total}")

        return precision, recall, f1


if __name__ == "__main__":
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

    #path to CSV relative path
    data_path = data_dir / "sales_data_final.csv"

    #load the entire dataset
    df = pd.read_csv(data_path)

    #enure 'order date' is in datetime format
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Order Date'] = df['Order Date'].fillna(df['Order Date'].min())

    #sort the dataframe by 'order date' in ascending order
    df_sorted = df.sort_values('Order Date').reset_index(drop=True)

    #define the split ratio
    train_ratio = 0.1
    total_rows = len(df_sorted)
    train_size = int(total_rows * train_ratio)
    test_size = total_rows - train_size
    TRAIN_SET_PATH = data_dir / "train_set.csv"
    TEST_SET_PATH = data_dir / "test_set.csv"

    #split into train and test sets
    train_set = df_sorted.iloc[:train_size].reset_index(drop=True)
    train_set.to_csv(TRAIN_SET_PATH, index=False)
    test_set = df_sorted.iloc[train_size:].reset_index(drop=True)
    test_set.to_csv(TEST_SET_PATH, index=False)

    #initialize the RecommendationModel with loaded=False to train the models
    model = RecommendationModel(loaded=False, train_df=train_set, test_df=test_set)

    CF_MODEL_PATH = models_dir / "collaborative_model.pkl"
    CB_MODELS_DIR = models_dir
    
    #save all models 
    model.save_all_models(cf_path=CF_MODEL_PATH, cb_directory=CB_MODELS_DIR)
