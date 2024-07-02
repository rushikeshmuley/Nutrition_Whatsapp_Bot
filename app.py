import os
import streamlit as st
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Initialize Twilio client
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

# Load or create the nutrition database
def load_nutrition_db():
    conn = sqlite3.connect('nutrition.db')
    try:
        df = pd.read_sql_query("SELECT * FROM nutrition", conn)
    except:
        # If the table doesn't exist, create it with sample data
        df = pd.DataFrame({
            'food': ['apple', 'banana', 'peach', 'spinach'],
            'calories': [52, 96, 39, 23],
            'protein': [0.3, 1.2, 0.9, 2.9],
            'carbs': [14, 22, 10, 3.6],
            'fat': [0.2, 0.2, 0.3, 0.4],
            'vitamins': ['C, B6', 'C, B6', 'C, A', 'A, C, K']
        })
        df.to_sql('nutrition', conn, index=False, if_exists='replace')
    conn.close()
    return df

nutrition_df = load_nutrition_db()

# Train a simple intent classifier
intents = [
    ('Which food is healthy for Adults?', 'healthy_food'),
    ('Does Apple have B12?', 'vitamin_query'),
    ('What does a healthy diet look like to you?', 'diet_advice'),
    ('What nutrition is there in Peach?', 'nutrition_query'),
    ('How many calories in banana?', 'nutrition_query'),
    ('Is spinach good for you?', 'health_benefits'),
]

X, y = zip(*intents)
clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
clf.fit(X, y)
joblib.dump(clf, 'intent_classifier.joblib')

# User profiling
user_profiles = {}

def get_user_profile(user_id):
    if user_id not in user_profiles:
        user_profiles[user_id] = {'history': [], 'preferences': {}}
    return user_profiles[user_id]

def update_user_profile(user_id, query, response):
    profile = get_user_profile(user_id)
    profile['history'].append((query, response))
    # Simple preference update based on mentioned foods
    for food in nutrition_df['food']:
        if food in query.lower():
            profile['preferences'][food] = profile['preferences'].get(food, 0) + 1

# Intent handling functions
def handle_healthy_food(query, user_id):
    profile = get_user_profile(user_id)
    preferred_foods = sorted(profile['preferences'], key=profile['preferences'].get, reverse=True)
    healthy_foods = [food for food in preferred_foods if nutrition_df[nutrition_df['food'] == food]['calories'].values[0] < 100]
    if healthy_foods:
        return f"Based on your preferences, some healthy foods for adults are: {', '.join(healthy_foods[:3])}. These are low in calories and rich in nutrients."
    else:
        return "Some healthy foods for adults include leafy greens, berries, lean proteins, and whole grains. These provide essential nutrients with relatively low calories."

def handle_vitamin_query(query, user_id):
    food = next((f for f in nutrition_df['food'] if f in query.lower()), None)
    if food:
        vitamins = nutrition_df[nutrition_df['food'] == food]['vitamins'].values[0]
        if 'B12' in vitamins:
            return f"Yes, {food} contains vitamin B12."
        else:
            return f"No, {food} does not contain significant amounts of vitamin B12. It does contain: {vitamins}."
    return "I couldn't find information about B12 in the specified food. Most plant-based foods don't naturally contain B12. It's mainly found in animal products and fortified foods."

def handle_diet_advice(query, user_id):
    return "A healthy diet typically includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. It's balanced, diverse, and tailored to individual needs. Remember to include different food groups in your meals and stay hydrated."

def handle_nutrition_query(query, user_id):
    food = next((f for f in nutrition_df['food'] if f in query.lower()), None)
    if food:
        info = nutrition_df[nutrition_df['food'] == food].iloc[0]
        return f"Nutrition info for {food}: Calories: {info['calories']}, Protein: {info['protein']}g, Carbs: {info['carbs']}g, Fat: {info['fat']}g, Vitamins: {info['vitamins']}"
    return "I'm sorry, I don't have information about that specific food in my database."

def handle_health_benefits(query, user_id):
    food = next((f for f in nutrition_df['food'] if f in query.lower()), None)
    if food:
        info = nutrition_df[nutrition_df['food'] == food].iloc[0]
        return f"{food.capitalize()} is nutritious. It's low in calories ({info['calories']}) and contains vitamins {info['vitamins']}. It's a good source of nutrients with minimal fat."
    return "I'm sorry, I don't have specific health benefit information for that food in my database."

intent_handlers = {
    'healthy_food': handle_healthy_food,
    'vitamin_query': handle_vitamin_query,
    'diet_advice': handle_diet_advice,
    'nutrition_query': handle_nutrition_query,
    'health_benefits': handle_health_benefits,
}

def generate_nutrition_response(query, user_id):
    # Classify intent
    intent = clf.predict([query])[0]
    
    # Handle intent
    if intent in intent_handlers:
        response = intent_handlers[intent](query, user_id)
    else:
        # Use Gemini AI for general queries
        prompt = f"""
        You are a nutrition expert chatbot. Provide accurate and helpful nutritional information for the following food item or question: {query}

        Include details such as:
        - Calories
        - Macronutrients (protein, carbs, fats)
        - Key vitamins and minerals
        - Health benefits or concerns

        Keep the response concise and easy to read on a mobile device.
        """
        response = model.generate_content(prompt).text
    
    # Update user profile
    update_user_profile(user_id, query, response)
    
    return response

# Streamlit UI
st.title("AI-Powered Food Nutrition Chatbot for WhatsApp")
st.write("Send a message to the WhatsApp number to get nutritional information.")

user_id = st.text_input("Enter your WhatsApp number:")
query = st.text_input("Enter your query:")

if st.button("Send Query"):
    if user_id and query:
        response = generate_nutrition_response(query, user_id)
        st.write(f"Response: {response}")
        
        # Send the response to the user via WhatsApp
        try:
            message = client.messages.create(
                body=response,
                from_='whatsapp:+<+16503186502>',  # Replace with your Twilio WhatsApp number
                to=f'whatsapp:+{user_id}'
            )
            st.write(f"Message sent to {user_id}")
        except TwilioRestException as e:
            st.error(f"Error sending message: {e}")
            
    else:
        st.write("Please enter your WhatsApp number and query.")
