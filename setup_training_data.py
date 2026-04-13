import os
import pandas as pd

# ==========================================
# עדכון שמות התיקיות לשמות שלך
# ==========================================
QUESTIONS_DIR = "data/questions"
ANSWERS_DIR = "data/solutions"
OUTPUT_FILE = "train.csv"

# ==========================================
# 1. סריקת הקבצים
# ==========================================
# בדיקה שהתיקיות בכלל קיימות כדי למנוע קריסה
if not os.path.exists(QUESTIONS_DIR) or not os.path.exists(ANSWERS_DIR):
    print(f"❌ Error: Folders not found!")
    print(f"   Looking for: {QUESTIONS_DIR}")
    print(f"   Looking for: {ANSWERS_DIR}")
    exit()

q_files = set(os.listdir(QUESTIONS_DIR))
a_files = set(os.listdir(ANSWERS_DIR))

# מוצאים רק את הקבצים המשותפים (שיש להם זוג)
valid_files = sorted(list(q_files.intersection(a_files)))

print(f"✅ Found {len(valid_files)} matching pairs (Question + Solution).")

# ==========================================
# 2. יצירת הטבלה
# ==========================================
data = []
for filename in valid_files:
    # אנחנו שמים "0" כברירת מחדל, אתה תשנה את זה אח"כ ידנית
    data.append({"filename": filename, "grade": 0})

# ==========================================
# 3. שמירה לקובץ
# ==========================================
if data:
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Created '{OUTPUT_FILE}'.")
    print("👉 ACTION REQUIRED: Open 'train.csv' now and fill in the real grades!")
else:
    print("❌ No matching files found! Check that image names match exactly (e.g. 'q1.jpg' in both folders).")