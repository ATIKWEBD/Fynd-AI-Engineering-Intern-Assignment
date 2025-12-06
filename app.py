"""
Fynd AI Engineering Internship Submission
Author: Sk Atik Ahemad
College: NIT Rourkela
Date: December 2025

Description:
    A dual-dashboard web application built with Streamlit.
    1. User Dashboard: Interfaces with customers to collect feedback and provide instant AI-generated replies.
    2. Admin Dashboard: Provides managers with summarized insights and actionable recommendations using Gemini 2.0 Flash.
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# Load environment variables for security (Best Practice)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") 

if not api_key:
    st.error("❌ API Key missing! Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# We use 'generation_config' to FORCE the model to return JSON.
# This makes the app crash-proof because the AI cannot "talk" extra text.
model = genai.GenerativeModel(
    'gemini-2.5-flash-lite',
    generation_config={"response_mime_type": "application/json"}
)

DATA_FILE = "reviews_data.csv"

# --- 2. BACKEND LOGIC ---

def get_data_store():
    """
    Reads the persistence layer (CSV).
    In a production environment, I would replace this with a SQL database connection.
    """
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Initialize schema with specific column names
        return pd.DataFrame(columns=[
            "Date", "Rating", "Review", 
            "AI_Response", "AI_Summary", "Recommended_Action"
        ])

def save_to_store(row_dict):
    """Appends a new interaction to the CSV data store."""
    df = get_data_store()
    new_row = pd.DataFrame([row_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def generate_ai_insights(review_text, rating):
    """
    Core AI Function:
    Uses a structured prompt to force the LLM to return valid JSON.
    Includes robust key extraction to prevent KeyErrors if AI uses slightly different names.
    """
    prompt = f"""
    You are an AI assistant for a retail store manager.
    Analyze this customer review.
    
    Customer Rating: {rating}/5
    Review Text: "{review_text}"
    
    Output a strictly valid JSON object with these EXACT 3 keys:
    1. "reply_to_customer": A polite, short reply to the customer.
    2. "internal_summary": A 5-word summary of the review.
    3. "manager_action": A concrete recommended action for the manager.
    """
    try:
        # Generate content (Model is already configured to return JSON)
        response = model.generate_content(prompt)
        
        # Parse the response
        data = json.loads(response.text)
        
        # Robust extraction: specific keys -> generic keys -> defaults
        # This prevents KeyErrors if the AI ignores specific key names
        return {
            "reply_to_customer": data.get("reply_to_customer", data.get("user_response", "Thank you for your feedback!")),
            "internal_summary": data.get("internal_summary", data.get("summary", "Summary unavailable")),
            "manager_action": data.get("manager_action", data.get("action", "Check manually"))
        }

    except Exception as e:
        print(f"❌ Error in AI generation: {e}")
        # Fallback logic to prevent app crash if AI fails
        return {
            "reply_to_customer": "Thank you for your feedback!",
            "internal_summary": "Error analyzing review",
            "manager_action": "Check manually"
        }

# --- 3. FRONTEND UI (Streamlit) ---

st.set_page_config(page_title="Fynd AI Feedback", layout="wide")
st.title("🤖 Fynd AI Feedback Loop")
st.markdown("### Designed by Sk Atik Ahemad")

# Using Tabs to separate User and Admin views cleanly
tab_user, tab_admin = st.tabs(["👤 Customer View", "🛠️ Admin Dashboard"])

# === TAB 1: USER VIEW ===
with tab_user:
    st.header("Leave a Review")
    st.write("Tell us about your experience!")
    
    # Input Widgets
    user_rating = st.slider("Rate your experience:", 1, 5, 5)
    user_review = st.text_area("Write your review here:", height=100)
    
    if st.button("Submit Review"):
        if user_review:
            with st.spinner("AI is analyzing your sentiment..."):
                # 1. Call AI
                ai_insight = generate_ai_insights(user_review, user_rating)
                
                # 2. Save Data
                # Note: We map the AI's keys to our Database columns here
                new_entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Rating": user_rating,
                    "Review": user_review,
                    "AI_Response": ai_insight['reply_to_customer'],
                    "AI_Summary": ai_insight['internal_summary'],
                    "Recommended_Action": ai_insight['manager_action']
                }
                save_to_store(new_entry)
                
                # 3. Feedback to User
                st.success("Submitted successfully!")
                st.info(f"**Store Response:** {ai_insight['reply_to_customer']}")
        else:
            st.warning("Please write some text before submitting.")

# === TAB 2: ADMIN VIEW ===
with tab_admin:
    st.header("Manager Dashboard")
    
    # Simple Authentication
    admin_password = st.text_input("Enter Admin Password", type="password")
    
    if admin_password == "fynd123":
        df = get_data_store()
        
        if not df.empty:
            # Analytics Section
            st.subheader("📊 Live Analytics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Reviews", len(df))
            with col2:
                avg = df["Rating"].mean()
                st.metric("Average Rating", f"{avg:.1f} / 5.0")
            
            st.bar_chart(df["Rating"].value_counts())

            # Insights Table
            st.subheader("📝 Actionable Insights")
            # Displaying only the most relevant columns for the manager
            st.dataframe(df[["Date", "Rating", "AI_Summary", "Recommended_Action", "Review"]], use_container_width=True)
            
            # Export Feature
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data", data=csv, file_name="fynd_reviews.csv")
        else:
            st.info("No reviews yet.")
    elif admin_password:
        st.error("Incorrect password.")