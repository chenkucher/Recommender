# Recommendation System

This repository contains a recommendation system that integrates collaborative filtering, content-based filtering, and a fine-tuned GPT model. The system is designed to analyze user purchase behavior and provide personalized product recommendations. It also includes a Telegram bot for user interaction.

## Project Structure

- `data_preprocess.py` - Processes raw data, adds new columns (`Customer_ID`, `Product_ID`, `behavior_rating`), and saves the processed data.
- `recommendation_model.py` - Builds the recommendation models (collaborative filtering and content-based filtering) and saves them for further use.
- `create_fine_tune_file.py` - Creates a `.jsonl` file for fine-tuning GPT-4o-mini using historical user interactions.
- `evaluate_models.py` - Evaluates the recommendation models using various metrics.
- `telegram_bot.py` - Runs a Telegram bot to interact with users and provide recommendations.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/chenkucher/Recommender.git
   cd Recommender
   ```
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Scripts

Follow this order to execute the scripts:

1. **Preprocess Data**
   ```
   python data_preprocess.py
   ```
   - This script processes the raw sales data and adds required features.

2. **Train and Save the Recommendation Models**
   ```
   python recommendation_model.py
   ```
   - Builds collaborative and content-based models and saves them.

3. **Create Fine-Tune File for GPT-4o-mini**
   ```
   python create_fine_tune_file.py
   ```
   - Generates `.jsonl` data for fine-tuning the GPT model.

4. **Evaluate the Models**
   ```
   python evaluate_models.py
   ```
   - Tests and evaluates the recommendation models.

5. **Run the Telegram Bot**
   ```
   python telegram_bot.py
   ```
   - Starts the Telegram bot for interacting with users.

## Environment Variables

Create a `.env` file in the project root with the following variables:

```ini
BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
FINE_TUNED_MODEL=your_finetuned_gpt_model
```

## File Paths
- Data should be placed in `data/`.
- Models are stored in `models/`.
- Fine-tune `.jsonl` files are saved in `data/`.


