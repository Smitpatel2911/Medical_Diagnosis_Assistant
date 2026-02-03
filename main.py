import pandas as pd
import numpy as np
import pickle
import sys
import time
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/final_model_dt.pkl'
SCALER_PATH = 'models/scaler.pkl'

# --- UTILS for "Human Touch" ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def slow_print(text, delay=0.01):
    """Prints text with a retro typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_banner():
    clear_screen()
    print("="*60)
    print("      🫀  HEART DISEASE DIAGNOSTIC ASSISTANT v1.0  🫀")
    print("="*60)
    print("  🏥  Master Level Artificial Intelligence System")
    print("  🛡️   Authorized Personnel Only")
    print("="*60 + "\n")

# --- INPUT HANDLER ---
def get_valid_input(prompt, type_=float, min_val=None, max_val=None, options=None):
    """Robust input function that handles errors gracefully."""
    while True:
        try:
            user_input = input(f"  👉 {prompt}: ")
            
            # Handle Options (e.g., Yes/No)
            if options:
                if user_input.lower() in options:
                    return options[user_input.lower()]
                else:
                    print(f"     ❌ Invalid option. Please choose: {', '.join(options.keys())}")
                    continue

            # Handle Numbers
            value = type_(user_input)
            if min_val is not None and value < min_val:
                print(f"     ⚠️  Value too low (Min: {min_val})")
                continue
            if max_val is not None and value > max_val:
                print(f"     ⚠️  Value too high (Max: {max_val})")
                continue
            return value

        except ValueError:
            print(f"     ❌ Please enter a valid {type_.__name__}.")

# --- MAIN APPLICATION ---
def main():
    print_banner()
    slow_print("🔄 Initializing System Modules...", delay=0.03)

    # 1. Load Resources
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        slow_print("✅ Model & Scaler Loaded Successfully.\n")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Missing files in {MODEL_PATH} or {SCALER_PATH}.")
        print("   Did you run the training notebooks and save the files?")
        return

    slow_print("📋 Please enter Patient Vitals below:\n")

    # 2. Collect Data
    # Continuous Features
    age = get_valid_input("Patient Age", int, 1, 120)
    trestbps = get_valid_input("Resting Blood Pressure (mm Hg)", int, 50, 250)
    chol = get_valid_input("Serum Cholesterol (mg/dl)", int, 100, 600)
    thalach = get_valid_input("Max Heart Rate Achieved", int, 50, 250)
    oldpeak = get_valid_input("ST Depression (Oldpeak)", float, 0.0, 10.0)
    
    # Categorical / Binary Features
    sex = get_valid_input("Sex (m/f)", str, options={'m': 1, 'f': 0})
    exang = get_valid_input("Exercise Induced Angina? (y/n)", str, options={'y': 1, 'n': 0})
    ca = get_valid_input("Major Vessels Colored by Fluoroscopy (0-3)", int, 0, 3)
    
    # Complex Categoricals (Need Mapping for One-Hot Encoding)
    print("\n  🔍 Chest Pain Type:")
    print("     0: Asymptomatic (Silent)")
    print("     1: Atypical Angina")
    print("     2: Non-anginal Pain")
    print("     3: Typical Angina")
    cp_input = get_valid_input("Select Type (0-3)", int, 0, 3)

    print("\n  📉 Slope of Peak Exercise:")
    print("     0: Downsloping")
    print("     1: Flat")
    print("     2: Upsloping")
    slope_input = get_valid_input("Select Slope (0-2)", int, 0, 2)

    print("\n  🧬 Thalassemia:")
    print("     0: Null/Unknown")
    print("     1: Fixed Defect")
    print("     2: Normal")
    print("     3: Reversable Defect")
    thal_input = get_valid_input("Select Type (0-3)", int, 0, 3)

    # 3. Preprocessing (The "Brain" Work)
    slow_print("\n🔄 Processing Vitals...", delay=0.02)

    # A. Feature Engineering (Risk Score)
    risk_score = 0
    if age > 60: risk_score += 1
    if trestbps > 140: risk_score += 1
    if chol > 240: risk_score += 1
    
    # B. Create DataFrame (Raw) - MUST match the scaler's expected columns
    # Note: We only scale specific columns
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    raw_data = pd.DataFrame([[age, trestbps, chol, thalach, oldpeak]], columns=cols_to_scale)
    
    # C. Apply Scaling
    scaled_values = scaler.transform(raw_data) # Returns an array
    
    # D. Prepare Final Input Vector (Order Must Match Training Columns Exactly!)
    # Let's construct the dictionary of all features expected by the model
    # Note: This list depends on YOUR specific One-Hot Encoding in Notebook 02.
    # We initialize all possibilities to 0.
    
    input_dict = {
        'age': scaled_values[0][0],
        'sex': sex,
        'trestbps': scaled_values[0][1],
        'chol': scaled_values[0][2],
        'fbs': 0, # Assuming 0 for simplicity or ask user
        'restecg': 0, # Assuming 0 or ask user
        'thalach': scaled_values[0][3],
        'exang': exang,
        'oldpeak': scaled_values[0][4],
        'ca': ca,
        'risk_score': risk_score,
        
        # One-Hot Encoded Columns (Initialize to 0)
        'cp_1': 0, 'cp_2': 0, 'cp_3': 0,
        'thal_1': 0, 'thal_2': 0, 'thal_3': 0,
        'slope_1': 0, 'slope_2': 0
    }

    # Set the specific One-Hot bits to 1 based on user input
    if cp_input in [1, 2, 3]: input_dict[f'cp_{cp_input}'] = 1
    if thal_input in [1, 2, 3]: input_dict[f'thal_{thal_input}'] = 1
    if slope_input in [1, 2]: input_dict[f'slope_{slope_input}'] = 1

    # Convert to DataFrame (Final Input)
    # Important: Ensure columns are in the correct order (Model will fail otherwise)
    # We use the model's feature_names_in_ attribute if available, or sort matching training
    try:
        expected_cols = model.feature_names_in_
        final_df = pd.DataFrame([input_dict])[expected_cols]
    except AttributeError:
        # Fallback if model doesn't store feature names (older sklearn versions)
        final_df = pd.DataFrame([input_dict])

    # 4. Prediction
    slow_print("🧠 Analyzing Patterns...", delay=0.05)
    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1] * 100

    # 5. Result Display
    print("\n" + "="*60)
    print("                  📢  DIAGNOSTIC REPORT  📢")
    print("="*60)
    
    if prediction == 1:
        print(f"\n  🔴 RESULT: POSITIVE FOR HEART DISEASE")
        print(f"  ⚠️  Confidence Level: {probability:.2f}%")
        print("  🚑  Recommendation: Immediate Clinical Consultation Required.")
    else:
        print(f"\n  🟢 RESULT: NEGATIVE (HEALTHY)")
        print(f"  ✅  Confidence Level: {100 - probability:.2f}%")
        print("  🏃  Recommendation: Maintain healthy lifestyle.")
        
    print("\n" + "="*60)
    print("  Disclaimer: This AI tool is for educational purposes only.")
    print("  It is NOT a substitute for professional medical advice.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()