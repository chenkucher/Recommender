import os
import openai
import telebot
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import threading 
import time
from pathlib import Path

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')  #telegram bot token
openai.api_key = os.getenv('OPENAI_API_KEY')
fine_tuned_model = os.getenv('FINE_TUNED_MODEL')

#get the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

#path to CSV relative path
csv_path = SCRIPT_DIR / "data" / "sales_data_final.csv"

df = pd.read_csv(csv_path)

#lock for thread-safety
df_lock = threading.Lock()


def calculate_top_categories(group):
    category_counter = Counter()
    top_categories_list = []
    
    for category in group['Product Category']:
        category_counter[category] += 1
        top_three = [cat for cat, count in category_counter.most_common(3)]
        top_categories_list.append(str(top_three))
    return top_categories_list

df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
if df['Order Date'].isnull().any():
    print("warning: some 'order date' entries could not be parsed and are set as NaT.")

df_sorted = df.sort_values(by=['Customer_ID', 'Order Date']).reset_index(drop=True)
df_sorted['Total Purchases'] = df_sorted.groupby('Customer_ID').cumcount() + 1

df_sorted['Top Categories'] = (
    df_sorted.groupby('Customer_ID')
             .apply(calculate_top_categories)
             .explode()
             .values
)

df = df_sorted

bot = telebot.TeleBot(BOT_TOKEN)

user_data = {}
user_state = {}
user_steps_flow = {}
user_cart = {}

recommendations_list = {}
recommendations_current_idx = {}

login_tries = {}

STATE_MAIN_MENU = "main_menu"
STATE_LOGIN_ASK_CUSTOMER_ID = "login_ask_customer_id"
STATE_REGISTER_FLOW = "register_flow"
STATE_RECOMMEND_FLOW = "recommend_flow_in_progress"
STATE_BEST_PRODUCTS_FLOW = "best_products_flow"
STATE_CHANGE_QUANTITY = "change_quantity"

STATE_CATEGORY_SEARCH = "category_search"  # to track user picking categories

STEPS_NO_ACCOUNT = ["ask_city", "ask_address"]
STEPS_FULL = ["ask_city", "ask_address"]

#helper functions
def construct_user_profile(row):
    customer_id = row['Customer_ID']
    if pd.notnull(row['City']):
        main_city = row['City'].strip()
    else:
        main_city = "unknown"

    
    total_purchases = row.get('Total Purchases', 0)
    if pd.isnull(total_purchases):
        total_purchases = 0
    
    top_categories_str = row.get('Top Categories', "")
    if pd.isnull(top_categories_str):
        top_categories_str = "none"
    
    user_profile_text = (
        f"user id: {customer_id}, main city: {main_city}, "
        f"total purchases: {total_purchases}, "
        f"top categories: {top_categories_str}"
    )
    return user_profile_text


def get_customer_data_from_csv(customer_id):
    global df
    try:
        customer_id_val = int(customer_id)
    except ValueError:
        customer_id_val = customer_id

    user_df = df[df['Customer_ID'] == customer_id_val]
    if user_df.empty:
        return None

    row = user_df.iloc[-1]
    
    user_profile_text = construct_user_profile(row)
    if pd.notnull(row['City']):
        row_city = row['City']
    else:
        row_city = "unknown city"

    if pd.notnull(row['Purchase Address']):
        row_addr = row['Purchase Address']
    else:
        row_addr = "unknown purchase address"

    
    return {
        "customer_id": str(customer_id_val),
        "user_profile_text": user_profile_text,
        "city": row_city,
        "purchase_address": row_addr
    }


def gpt_recommend_chat(fine_tuned_model, user_profile_text):
    prompt_lines = [
        "User Profile:",
        user_profile_text,
        "\nRecommend 5 products this user might be interested in:"
    ]
    user_message_content = "\n".join(prompt_lines).strip()

    response = openai.ChatCompletion.create(
        model=fine_tuned_model,
        messages=[{"role": "user", "content": user_message_content}],
        max_tokens=150,
        temperature=0.7
    )
    assistant_message = response["choices"][0]["message"]["content"].strip()
    return assistant_message


def fetch_product_info(product_name, idx=None):
    global df
    matching_rows = df[df["Product"] == product_name]

    if not matching_rows.empty:
        row_match = matching_rows.iloc[0]

        #extract values with default
        product_id = row_match.get("Product_ID", None)
        price = row_match.get("Price Each", 0.0)
        category = row_match.get("Product Category", "null")

        #handle missing values
        product_id = f"{idx+1}" if pd.isnull(product_id) and idx is not None else product_id or "RECO"
        price = 0.0 if pd.isnull(price) else price
        category = "null" if pd.isnull(category) else category
    else:
        product_id = f"RECO_{idx+1}" if idx is not None else "RECO"
        price = 0.0
        category = "null"

    return product_id, price, category




def is_user_registered(chat_id):
    if chat_id not in user_data:
        return False
    if "customer_id" not in user_data[chat_id]:
        return False
    if "user_profile_text" not in user_data[chat_id]:
        return False
    return True


def generate_new_customer_id():
    global df
    cust_rows = df["Customer_ID"].dropna()

    if cust_rows.empty:
        next_num = 1
    else:
        try:
            max_num = cust_rows.astype(int).max()
            next_num = max_num + 1
        except ValueError:
            max_num = 0
            for cid in cust_rows:
                try:
                    numeric_val = int(cid)
                    if numeric_val > max_num:
                        max_num = numeric_val
                except ValueError:
                    pass
            next_num = max_num + 1

    new_id = str(next_num)
    return new_id


def append_user_action_to_csv(chat_id, product_name, product_category, quantity_ordered, price_each, product_id):


    global df, csv_path

    W_P = 5
    W_C = 1
    W_Q = 0.2

    if df["Order ID"].empty:
        new_order_id = 100000
    else:
        new_order_id = df["Order ID"].max() + 1

    now = datetime.now()
    order_date = pd.to_datetime(now)
    order_month = order_date.month
    order_hour = order_date.hour

    if 5 <= order_hour < 12:
        time_of_day = "morning"
    elif 12 <= order_hour < 18:
        time_of_day = "afternoon"
    elif 18 <= order_hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    sales_value = float(price_each) * float(quantity_ordered)

    customer_id = user_data[chat_id].get("customer_id", "0")
    city = user_data[chat_id].get("city", "unknown city")
    purchase_address = user_data[chat_id].get("purchase_address", "unknown purchase address")

    new_row = {
        "Column1": None,
        "Order ID": new_order_id,
        "Product Category": product_category,
        "Product": product_name,
        "Quantity Ordered": quantity_ordered,
        "Price Each": price_each,
        "Order Date": order_date,
        "Purchase Address": purchase_address,
        "Month": order_month,
        "Sales": sales_value,
        "City": city,
        "Hour": order_hour,
        "Time of Day": time_of_day,
        "Customer_ID": customer_id,
        "Product_ID": product_id,
        "behavior_rating": 0.0,
        "Total Purchases": 0,
        "Top Categories": ""
    }

    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    df.sort_values(by=["Customer_ID", "Order Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cust_product_count = defaultdict(lambda: defaultdict(int))
    cust_cat_count = defaultdict(lambda: defaultdict(int))

    df["behavior_rating"] = 0.0

    for i, row in df.iterrows():
        c_id = row["Customer_ID"]
        p_id = row["Product_ID"]
        cat = row["Product Category"]
        qty = row["Quantity Ordered"]

        product_ct = cust_product_count[c_id][p_id]
        cat_ct = cust_cat_count[c_id][cat]

        rating = (W_P * product_ct
                  + W_C * cat_ct
                  + W_Q * qty)

        df.at[i, "behavior_rating"] = rating

        cust_product_count[c_id][p_id] += 1
        cust_cat_count[c_id][cat] += 1

    df["Total Purchases"] = df.groupby("Customer_ID").cumcount() + 1

    def compute_top_categories(group):
        cat_counter = Counter()
        result = []
        for category in group["Product Category"]:
            cat_counter[category] += 1
            
            top_three = []
            for category, _ in cat_counter.most_common(3):
                top_three.append(category)

            result.append(str(top_three))
        return result

    df["Top Categories"] = (
        df.groupby("Customer_ID")
          .apply(compute_top_categories)
          .explode()
          .values
    )

    df.to_csv(csv_path, index=False)

    print(f"[LOG] appended purchase row for user {customer_id} -> '{product_name}' (qty={quantity_ordered}).")


def format_product_message(product_name, price, category, idx):
    return (
        f"recommended product #{idx+1}: *{product_name}*\n"
        f"category: {category}\n"
        f"price: ${price}\n"
        "short description: great quality and a customer-favorite!"
    )


def calculate_cart_total(cart_items):
    total = 0.0
    for item in cart_items:
        total += item["price"] * item["quantity"]
    return round(total, 2)


def get_top_5_most_sold_products_last_7_days():
    global df
    df_temp = df.copy()
    df_temp = df_temp[df_temp["Order Date"].notnull()]
    
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    recent_df = df_temp[df_temp["Order Date"] >= seven_days_ago]
    if recent_df.empty:
        return []

    sales_counts = (
        recent_df
        .groupby("Product")["Quantity Ordered"]
        .sum()
        .sort_values(ascending=False)
    )
    top_5_products = sales_counts.head(5).index.tolist()
    return top_5_products


#telegram bot handkers
@bot.message_handler(commands=['start'])
def start_conversation(message):
    chat_id = message.chat.id

    user_data[chat_id] = {}
    user_cart[chat_id] = []
    user_state[chat_id] = STATE_MAIN_MENU
    login_tries[chat_id] = 0

    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_login = telebot.types.InlineKeyboardButton("login", callback_data="mainmenu_login")
    btn_register = telebot.types.InlineKeyboardButton("register", callback_data="mainmenu_register")
    btn_best_products = telebot.types.InlineKeyboardButton("check our best sellers", callback_data="mainmenu_best")
    btn_search_by_category = telebot.types.InlineKeyboardButton("search by category", callback_data="mainmenu_search_by_category")
    keyboard.add(btn_login, btn_register)
    keyboard.add(btn_best_products)
    keyboard.add(btn_search_by_category)

    bot.send_message(chat_id, "welcome to our store! please choose:", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data.startswith("mainmenu_"))
def handle_main_menu_actions(call):
    chat_id = call.message.chat.id
    data = call.data

    if data in ["mainmenu_login", "mainmenu_login_from_checkout"]:
        user_state[chat_id] = STATE_LOGIN_ASK_CUSTOMER_ID
        bot.edit_message_text(
            text="please enter your customer id:",
            chat_id=chat_id,
            message_id=call.message.message_id
        )

    elif data in ["mainmenu_register", "mainmenu_register_from_checkout"]:
        user_steps_flow[chat_id] = STEPS_NO_ACCOUNT
        user_state[chat_id] = STEPS_NO_ACCOUNT[0]  #"ask_city"

        new_cust_id = generate_new_customer_id()
        user_data[chat_id] = {
            "customer_id": new_cust_id,
            "city": "",
            "purchase_address": "",
            "user_profile_text": ""
        }

        bot.edit_message_text(
            text=(
                "we have assigned you a new customer id: "
                f"{new_cust_id}\n\n"
                "what city are you from?"
            ),
            chat_id=chat_id,
            message_id=call.message.message_id
        )

    elif data == "mainmenu_best":
        top_products = get_top_5_most_sold_products_last_7_days()
        if not top_products:
            bot.edit_message_text(
                text="no best-sellers found in the last 7 days!",
                chat_id=chat_id,
                message_id=call.message.message_id
            )
            bot.answer_callback_query(call.id)
            return

        recommendations_list[chat_id] = top_products
        recommendations_current_idx[chat_id] = 0
        user_state[chat_id] = STATE_BEST_PRODUCTS_FLOW

        bot.edit_message_text(
            text="check out our hottest products from the last 7 days!",
            chat_id=chat_id,
            message_id=call.message.message_id
        )
        show_next_recommendation(chat_id)

    elif data == "mainmenu_search_by_category":
        # Start the category search flow (available pre-login)
        user_state[chat_id] = STATE_CATEGORY_SEARCH
        user_data[chat_id]["selected_categories"] = set()  # track chosen categories
        bot.edit_message_text(
            text="choose up to 3 categories below:",
            chat_id=chat_id,
            message_id=call.message.message_id
        )
        send_category_selection_menu(chat_id)

    bot.answer_callback_query(call.id)


#category selection menu and callback handling

CATEGORIES_LIST = [
    "Audio Devices",
    "Batteries",
    "Charging Cables",
    "Entertainment Devices",
    "Home Appliances",
    "Laptops and Computers",
    "Monitors",
    "Phones and Accessories"
]

def send_category_selection_menu(chat_id, edit_message_id=None):
    """
    Show an inline keyboard with all categories. Mark selected with a check,
    plus 'submit' and 'cancel' at the bottom.
    """
    selected = user_data[chat_id].get("selected_categories", set())

    keyboard = telebot.types.InlineKeyboardMarkup()

    #build a button for each category (selection)
    for category in CATEGORIES_LIST:
        is_selected = (category in selected)
        label = f"✓ {category}" if is_selected else category
        callback_data = f"catsearch_toggle_{category}"
        keyboard.add(
            telebot.types.InlineKeyboardButton(label, callback_data=callback_data)
        )

    #submit/cancel buttons
    btn_submit = telebot.types.InlineKeyboardButton("submit", callback_data="catsearch_submit")
    btn_cancel = telebot.types.InlineKeyboardButton("cancel", callback_data="catsearch_cancel")
    keyboard.add(btn_submit, btn_cancel)

    if edit_message_id:
        bot.edit_message_text(
            text="choose up to 3 categories below:",
            chat_id=chat_id,
            message_id=edit_message_id,
            reply_markup=keyboard
        )
    else:
        bot.send_message(
            chat_id,
            "choose up to 3 categories below:",
            reply_markup=keyboard
        )


@bot.callback_query_handler(func=lambda call: call.data.startswith("catsearch_"))
def handle_category_search_callbacks(call):
    """
    Handler for toggling categories, submit, or cancel in the "search by category" flow.
    """
    chat_id = call.message.chat.id
    data = call.data
    user_state_current = user_state.get(chat_id)

    if user_state_current != STATE_CATEGORY_SEARCH:
        bot.answer_callback_query(call.id, "No active category selection.")
        return

    if data == "catsearch_cancel":
        bot.answer_callback_query(call.id, "Canceled category search.")
        #return to main menu
        user_state[chat_id] = STATE_MAIN_MENU
        bot.edit_message_text(
            text="Category search canceled. Use /start again or choose another action.",
            chat_id=chat_id,
            message_id=call.message.message_id
        )
        return

    if data == "catsearch_submit":
        selected = user_data[chat_id].get("selected_categories", set())
        if not selected:
            bot.answer_callback_query(call.id, "You must select at least 1 category.")
            return
        #build a mock user profile
        categories_str = ", ".join(selected)
        mock_user_profile = (
            f"user id: catsearch, main city: unknown, total purchases: 0, "
            f"top categories: {categories_str}"
        )
        #GPT recommendation with that profile
        recommendations_text = gpt_recommend_chat(
            fine_tuned_model=fine_tuned_model,
            user_profile_text=mock_user_profile
        )
        #split lines and remove empty ones
        lines = []

        for l in recommendations_text.split("\n"):
            stripped_line = l.strip()
            if stripped_line:
                lines.append(stripped_line)

        #split by comma if only one line
        if len(lines) == 1 and "," in lines[0]:
            split_lines = []

            for item in lines[0].split(","):
                stripped_item = item.strip()
                if stripped_item:
                    split_lines.append(stripped_item)

            lines = split_lines


        recommendations_list[chat_id] = lines
        recommendations_current_idx[chat_id] = 0
        user_state[chat_id] = STATE_RECOMMEND_FLOW

        if not lines:
            bot.edit_message_text(
                text="No recommendations found for these categories. Sorry!",
                chat_id=chat_id,
                message_id=call.message.message_id
            )
            return

        bot.edit_message_text(
            text=f"Found some recommendations for categories: {categories_str}.\nLet's see them!",
            chat_id=chat_id,
            message_id=call.message.message_id
        )
        show_next_recommendation(chat_id)
        return

    if data.startswith("catsearch_toggle_"):
        prefix, cat_name = data.split("catsearch_toggle_")
        cat_name = cat_name.strip()
        selected = user_data[chat_id].get("selected_categories", set())

        if cat_name in selected:
            #remove it
            selected.remove(cat_name)
            bot.answer_callback_query(call.id, f"Removed {cat_name}")
        else:
            #check if we already have 3 selected
            if len(selected) >= 3:
                bot.answer_callback_query(call.id, "You can select a maximum of 3 categories.")
                return
            #otherwise, add
            selected.add(cat_name)
            bot.answer_callback_query(call.id, f"Selected {cat_name}")

        user_data[chat_id]["selected_categories"] = selected
        #re-render menu
        send_category_selection_menu(chat_id, edit_message_id=call.message.message_id)


@bot.message_handler(commands=['recommend'])
def send_recommendations(message):
    chat_id = message.chat.id
    if not is_user_registered(chat_id):
        bot.reply_to(message, "please /start and login or register first.")
        return

    user_profile_text = user_data[chat_id].get("user_profile_text", "")
    if not user_profile_text:
        bot.reply_to(message, "your profile is incomplete. please /start or register.")
        return

    recommendations_text = gpt_recommend_chat(
        fine_tuned_model=fine_tuned_model,
        user_profile_text=user_profile_text
    )

    #split lines and remove empty ones
    lines = []

    for l in recommendations_text.split("\n"):
        stripped_line = l.strip()
        if stripped_line:
            lines.append(stripped_line)

    #split by comma
    if len(lines) == 1 and "," in lines[0]:
        split_lines = []

        for item in lines[0].split(","):
            stripped_item = item.strip()
            if stripped_item:
                split_lines.append(stripped_item)

        lines = split_lines


    recommendations_list[chat_id] = lines
    recommendations_current_idx[chat_id] = 0
    user_state[chat_id] = STATE_RECOMMEND_FLOW

    if not lines:
        bot.reply_to(message, "no recommendations found. sorry!")
        return

    bot.reply_to(
        message,
        "here are some products you might love. let’s go through them one by one!"
    )
    show_next_recommendation(chat_id)


def show_next_recommendation(chat_id):
    idx = recommendations_current_idx.get(chat_id, 0)
    items = recommendations_list.get(chat_id, [])

    if idx >= len(items):
        bot.send_message(
            chat_id,
            "that's all of the recommended products!\n"
            "if you need more, type /recommend again or /checkout to view your cart."
        )
        user_state.pop(chat_id, None)
        return

    product_name = items[idx]
    product_id, price, category = fetch_product_info(product_name, idx)
    msg_text = format_product_message(product_name, price, category, idx)

    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_more_info = telebot.types.InlineKeyboardButton("more info", callback_data=f"info_{idx}")
    btn_cart = telebot.types.InlineKeyboardButton("add to cart", callback_data=f"cart_{idx}")
    btn_skip = telebot.types.InlineKeyboardButton("skip", callback_data=f"skip_{idx}")

    keyboard.add(btn_more_info, btn_cart, btn_skip)
    bot.send_message(chat_id, msg_text, parse_mode="Markdown", reply_markup=keyboard)


@bot.message_handler(commands=['view_cart'])
def view_cart(message):
    chat_id = message.chat.id
    cart_items = user_cart.get(chat_id, [])

    if not cart_items:
        bot.send_message(chat_id, "your cart is empty. add some items with /recommend!")
        return

    cart_lines = []
    for i, item in enumerate(cart_items):
        line = f"{i+1}. {item['product_name']} (x{item['quantity']}) - ${item['price']} each"
        cart_lines.append(line)

    total_price = calculate_cart_total(cart_items)
    msg_text = (
        "🛒 *your cart:*\n"
        + "\n".join(cart_lines)
        + f"\n\ntotal: ${total_price}\n"
        "what would you like to do next?"
    )

    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_checkout = telebot.types.InlineKeyboardButton("proceed to checkout", callback_data="viewcart_checkout")
    btn_recommend = telebot.types.InlineKeyboardButton("continue shopping", callback_data="viewcart_recommend")
    keyboard.add(btn_checkout, btn_recommend)

    bot.send_message(chat_id, msg_text, parse_mode="Markdown", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data.startswith("viewcart_"))
def handle_viewcart_buttons(call):
    chat_id = call.message.chat.id
    data = call.data

    bot.answer_callback_query(call.id)

    if data == "viewcart_checkout":
        checkout(call.message)
    elif data == "viewcart_recommend":
        send_recommendations(call.message)


@bot.callback_query_handler(func=lambda call: True)
def handle_inline_actions(call):
    chat_id = call.message.chat.id
    data = call.data

    #changing qty
    if data.startswith("chgqty_"):
        parts = data.split("_")
        if len(parts) != 2:
            bot.answer_callback_query(call.id, "invalid action.")
            return
        try:
            item_index = int(parts[1])
        except ValueError:
            bot.answer_callback_query(call.id, "invalid index.")
            return
        
        user_state[chat_id] = STATE_CHANGE_QUANTITY
        user_data[chat_id]["quantity_update_index"] = item_index
        bot.answer_callback_query(call.id)
        bot.send_message(chat_id, f"please enter the new quantity for item #{item_index + 1}:")
        return

    #remove_x
    elif data.startswith("remove_"):
        parts = data.split("_")
        if len(parts) != 2:
            bot.answer_callback_query(call.id, "invalid action.")
            return
        try:
            item_index = int(parts[1])
        except ValueError:
            bot.answer_callback_query(call.id, "invalid item index.")
            return

        if 0 <= item_index < len(user_cart.get(chat_id, [])):
            removed_item = user_cart[chat_id].pop(item_index)
            bot.answer_callback_query(call.id, f"removed '{removed_item['product_name']}' from your cart.")
            checkout(call.message)
        else:
            bot.answer_callback_query(call.id, "invalid item index.")
        return

    #confirm / cancel checkout
    elif data in ["confirm_checkout", "cancel_checkout"]:
        bot.answer_callback_query(call.id)
        handle_checkout_flow(call, data)
        return

    #if user is picking categories, that is handled by catsearch_ above. 
    #if user is in main menu or login:
    if user_state.get(chat_id) in [STATE_MAIN_MENU, STATE_LOGIN_ASK_CUSTOMER_ID, STATE_CATEGORY_SEARCH]:
        bot.answer_callback_query(call.id, "please finish your current action first or select an option.")
        return

    #recommendation or best product flows
    if user_state.get(chat_id) in [STATE_RECOMMEND_FLOW, STATE_BEST_PRODUCTS_FLOW]:
        parts = data.split("_")
        if len(parts) != 2:
            bot.answer_callback_query(call.id, "invalid action.")
            return
        action, idx_str = parts
        try:
            idx = int(idx_str)
        except ValueError:
            bot.answer_callback_query(call.id, "invalid product index.")
            return

        items = recommendations_list.get(chat_id, [])
        if not (0 <= idx < len(items)):
            bot.answer_callback_query(call.id, "index out of range.")
            return

        product_name = items[idx]
        product_id, price, category = fetch_product_info(product_name, idx)

        if action == "skip":
            bot.answer_callback_query(call.id, f"skipped product: {product_name}")
            recommendations_current_idx[chat_id] += 1
            show_next_recommendation(chat_id)

        elif action == "info":
            extended_info_text = (
                f"*{product_name}*\n"
                f"category: {category}\n"
                f"price: ${price}\n"
                "_detailed description:_\n"
                "this is a highly recommended product with excellent user reviews.\n"
                "shipping within 3-5 business days."
            )
            bot.answer_callback_query(call.id, "showing more info...")
            bot.send_message(chat_id, extended_info_text, parse_mode="Markdown")

        elif action == "cart":
            user_cart[chat_id].append({
                "product_id": product_id,
                "product_name": product_name,
                "price": price,
                "category": category,
                "quantity": 1
            })
            bot.answer_callback_query(call.id, f"added '{product_name}' to your cart!")
            recommendations_current_idx[chat_id] += 1
            show_next_recommendation(chat_id)
        else:
            bot.answer_callback_query(call.id, "invalid button action.")
    else:
        bot.answer_callback_query(call.id, "no active flow.")


@bot.message_handler(commands=['checkout'])
def checkout(message):
    chat_id = message.chat.id
    cart_items = user_cart.get(chat_id, [])

    if not cart_items:
        bot.send_message(chat_id, "your cart is empty. add some items with /recommend!")
        return

    if not is_user_registered(chat_id):
        bot.send_message(
            chat_id,
            "you need to be registered or logged in to proceed with the purchase.\n"
            "please choose one of the following options."
        )
        keyboard = telebot.types.InlineKeyboardMarkup()
        btn_login = telebot.types.InlineKeyboardButton("login", callback_data="mainmenu_login_from_checkout")
        btn_register = telebot.types.InlineKeyboardButton("register", callback_data="mainmenu_register_from_checkout")
        keyboard.add(btn_login, btn_register)
        bot.send_message(chat_id, "select an option:", reply_markup=keyboard)
        return

    cart_lines = []
    for i, item in enumerate(cart_items):
        line = f"{i+1}. {item['product_name']} (x{item['quantity']}) - ${item['price']} each"
        cart_lines.append(line)

    total_price = calculate_cart_total(cart_items)
    discount_code = "save10"
    discount_applied = round(total_price * 0.1, 2)
    final_price = round(total_price - discount_applied, 2)

    summary_text = (
        "🛒 *checkout summary:*\n"
        + "\n".join(cart_lines)
        + f"\n\nsubtotal: ${total_price:.2f}\n"
        f"discount code: {discount_code}\n"
        f"discount amount: -${discount_applied:.2f}\n"
        f"final price: ${final_price:.2f}\n\n"
        "you can adjust quantities or remove items below, then confirm your purchase."
    )

    keyboard = telebot.types.InlineKeyboardMarkup()
    for i, item in enumerate(cart_items):
        btn_chgqty = telebot.types.InlineKeyboardButton(
            text=f"change qty (item {i+1})",
            callback_data=f"chgqty_{i}"
        )
        btn_remove = telebot.types.InlineKeyboardButton(
            text=f"remove (item {i+1})",
            callback_data=f"remove_{i}"
        )
        keyboard.add(btn_chgqty, btn_remove)

    btn_confirm = telebot.types.InlineKeyboardButton("confirm purchase", callback_data="confirm_checkout")
    btn_cancel = telebot.types.InlineKeyboardButton("cancel", callback_data="cancel_checkout")
    keyboard.add(btn_confirm, btn_cancel)

    bot.send_message(chat_id, summary_text, parse_mode="Markdown", reply_markup=keyboard)


def process_checkout(chat_id, message_id):
    try:
        bot.edit_message_text(
            text="🕒 We are processing your request...",
            chat_id=chat_id,
            message_id=message_id
        )
        time.sleep(15)  #simulate processing time

        with df_lock:
            for item in user_cart[chat_id]:
                append_user_action_to_csv(
                    chat_id,
                    item["product_name"],
                    item["category"],
                    item["quantity"],
                    item["price"],
                    item["product_id"]
                )
            user_cart[chat_id] = []

        bot.edit_message_text(
            text="✅ Purchase confirmed! Thank you for shopping with us.\n",
            chat_id=chat_id,
            message_id=message_id
        )

    except Exception as e:
        bot.edit_message_text(
            text="❌ An error occurred while processing your purchase. Please try again later.",
            chat_id=chat_id,
            message_id=message_id
        )
        print(f"Error processing checkout for chat_id {chat_id}: {e}")


def handle_checkout_flow(call, data):
    chat_id = call.message.chat.id

    if data == "confirm_checkout":
        threading.Thread(target=process_checkout, args=(chat_id, call.message.message_id)).start()

    elif data == "cancel_checkout":
        bot.edit_message_text(
            text=(
                "checkout canceled. your items remain in your cart.\n"
                "feel free to /checkout again anytime."
            ),
            chat_id=chat_id,
            message_id=call.message.message_id
        )


@bot.message_handler(commands=['catsearch'])
def category_search_command(message):

    chat_id = message.chat.id
    if not is_user_registered(chat_id):
        bot.send_message(chat_id, "please /start and login or register first.")
        return
    user_state[chat_id] = STATE_CATEGORY_SEARCH
    user_data[chat_id]["selected_categories"] = set()
    bot.send_message(chat_id, "Choose up to 3 categories below:")
    send_category_selection_menu(chat_id)


@bot.message_handler(func=lambda msg: True)
def handle_user_responses(message):
    chat_id = message.chat.id
    text = message.text.strip()

    #login flow
    if user_state.get(chat_id) == STATE_LOGIN_ASK_CUSTOMER_ID:
        customer_id_input = text
        csv_data = get_customer_data_from_csv(customer_id_input)
        if csv_data:
            user_data[chat_id] = {
                "customer_id": csv_data["customer_id"],
                "user_profile_text": csv_data["user_profile_text"],
                "city": csv_data["city"],
                "purchase_address": csv_data["purchase_address"]
            }
            user_state.pop(chat_id, None)
            user_steps_flow.pop(chat_id, None)
            login_tries[chat_id] = 0

            bot.send_message(
                chat_id,
                "great, we found your account and loaded your profile!\n"
                f"city: {csv_data['city']}\n"
                f"last known address: {csv_data['purchase_address']}\n\n"
                "you can type /recommend for suggestions, /view_cart to see your cart, or /catsearch to search by category."
            )
        else:
            login_tries[chat_id] += 1
            if login_tries[chat_id] < 3:
                bot.send_message(
                    chat_id,
                    f"customer id not found. please try again ({3 - login_tries[chat_id]} tries left):"
                )
            else:
                user_state[chat_id] = STATE_MAIN_MENU
                bot.send_message(
                    chat_id,
                    "you tried 3 times with no success. please register or check our best sellers."
                )
                keyboard = telebot.types.InlineKeyboardMarkup()
                btn_register = telebot.types.InlineKeyboardButton("register", callback_data="mainmenu_register")
                btn_best_products = telebot.types.InlineKeyboardButton("check our best sellers", callback_data="mainmenu_best")
                keyboard.add(btn_register, btn_best_products)
                bot.send_message(chat_id, "choose an option:", reply_markup=keyboard)
        return

    #registration / profile flow
    if chat_id in user_steps_flow:
        flow = user_steps_flow[chat_id]
        state = user_state[chat_id]
        if state not in flow:
            bot.send_message(chat_id, "please use /recommend or /start.")
            return

        current_index = flow.index(state)

        if state == "ask_city":
            user_data[chat_id]["city"] = text.strip()
            user_state[chat_id] = flow[current_index + 1]  #ask_address
            bot.send_message(chat_id, "please enter your full address:")
            return

        elif state == "ask_address":
            user_data[chat_id]["purchase_address"] = text.strip()

            new_cust_id = user_data[chat_id]["customer_id"]
            user_profile_text = (
                f"user id: {new_cust_id}, main city: {user_data[chat_id]['city']}, "
                f"total purchases: 0, "
                "top categories: none"
            )
            user_data[chat_id]["user_profile_text"] = user_profile_text

            user_state.pop(chat_id, None)
            user_steps_flow.pop(chat_id, None)

            bot.send_message(
                chat_id,
                "great! your profile is complete.\n"
                "you can now type /recommend to get personalized recommendations, /view_cart to view your cart, or /catsearch to search by category!"
            )
            return

        else:
            bot.send_message(chat_id, "please use /recommend or /start.")
        return

    #new quantity update in checkout
    if user_state.get(chat_id) == STATE_CHANGE_QUANTITY:
        if not text.isdigit():
            bot.send_message(chat_id, "please enter a valid number for the quantity.")
            return

        new_qty = int(text)
        if new_qty < 1:
            bot.send_message(chat_id, "quantity cannot be 0 or negative.")
            return

        item_index = user_data[chat_id].get("quantity_update_index")
        if item_index is None:
            bot.send_message(chat_id, "something went wrong. please try /checkout again.")
            user_state.pop(chat_id, None)
            return

        if 0 <= item_index < len(user_cart[chat_id]):
            user_cart[chat_id][item_index]["quantity"] = new_qty
            bot.send_message(chat_id, f"quantity updated to {new_qty}.")
        else:
            bot.send_message(chat_id, "invalid item index.")

        user_state.pop(chat_id, None)
        user_data[chat_id].pop("quantity_update_index", None)
        checkout(message)
        return

    #fallback
    bot.send_message(chat_id, "please choose /start to begin or /recommend if you've logged in.")


if __name__ == "__main__":
    print("bot is running...")
    bot.infinity_polling()
