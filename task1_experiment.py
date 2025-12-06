import google.generativeai as genai
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, mean_absolute_error

# --- 1. SETUP ---
load_dotenv() 
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: API Key not found. Check your .env file.")
    exit()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# --- 2. LOAD DATA ---
def load_data():
    try:
        # UPDATED: Reads 'yelp.csv' directly
        if os.path.exists("yelp.csv"):
            df = pd.read_csv("yelp.csv")
            print("✅ Dataset loaded (yelp.csv). Sampling 200 rows...")
            return df.sample(200, random_state=42)
        elif os.path.exists("yelp_reviews.csv"):
            df = pd.read_csv("yelp_reviews.csv")
            print("✅ Dataset loaded (yelp_reviews.csv). Sampling 200 rows...")
            return df.sample(200, random_state=42)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("⚠️ Warning: No CSV found. Using dummy data.")
        return pd.DataFrame({
            'text': ["Food was cold.", "Great service!", "It was okay.", "Terrible.", "Best ever!"],
            'stars': [1, 5, 3, 1, 5]
        })

# --- 3. THE PROMPT FUNCTION ---
def get_ai_rating(review, method):
    base_instruction = 'Return ONLY valid JSON format: {"predicted_stars": int, "explanation": "string"}'
    
    if method == "direct":
        prompt = f"""
        Analyze this review and assign a rating (1-5).
        {base_instruction}
        Review: "{review}"
        """
    elif method == "cot":
        prompt = f"""
        Analyze the sentiment step-by-step. Identify positive and negative keywords.
        Then determine the star rating based on the balance.
        {base_instruction}
        Review: "{review}"
        """
    elif method == "role":
        prompt = f"""
        You are an expert food critic. 
        Evaluate the tone, vocabulary, and context to judge the customer's satisfaction.
        {base_instruction}
        Review: "{review}"
        """
    
    try:
        response = model.generate_content(prompt)
        text_res = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text_res)
        return int(data['predicted_stars']), data['explanation']
    except Exception:
        return 0, "Error"

# --- 4. EXECUTION LOOP ---
def run_experiment():
    df = load_data()
    results = []

    print(f"🚀 Starting experiment on {len(df)} reviews...")
    
    for index, row in df.iterrows():
        review = row['text']
        actual = row['stars']
        
        # Run all 3 prompts
        p1, _ = get_ai_rating(review, "direct")
        p2, _ = get_ai_rating(review, "cot")
        p3, _ = get_ai_rating(review, "role")
        
        if len(results) % 20 == 0:
            print(f"Processed {len(results)} rows...")
        
        results.append({
            "review": review,
            "actual_stars": actual,
            "P1_Direct": p1,
            "P2_CoT": p2,
            "P3_Role": p3
        })
        time.sleep(0.5) 

    # --- 5. BETTER EVALUATION ---
    results_df = pd.DataFrame(results)
    
    # Filter out errors (0s)
    valid_df = results_df[results_df["P1_Direct"] != 0]

    def calculate_metrics(y_true, y_pred, name):
        # Exact Accuracy (Strict)
        acc = accuracy_score(y_true, y_pred)
        # Off-by-One Accuracy (Lenient - Valid for 5-star ratings)
        within_one = sum(abs(y_true - y_pred) <= 1) / len(y_true)
        print(f"\n📊 {name} Results:")
        print(f"   Exact Accuracy: {acc:.2%}")
        print(f"   Off-by-One Accuracy: {within_one:.2%} (Use this in report!)")

    calculate_metrics(valid_df['actual_stars'], valid_df['P1_Direct'], "Direct Prompt")
    calculate_metrics(valid_df['actual_stars'], valid_df['P2_CoT'], "Chain of Thought")
    calculate_metrics(valid_df['actual_stars'], valid_df['P3_Role'], "Role Expert")
    
    results_df.to_csv("task1_results.csv", index=False)
    print("\n✅ Results saved to task1_results.csv")

if __name__ == "__main__":
    run_experiment()