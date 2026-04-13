import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================================
# הגדרות
# ==========================================
INPUT_FILE = "training_features.csv"
MODEL_FILE = "grader_model.pth"
SCALER_FILE = "scaler.pkl"
GRAPH_FILE = "loss_curve.png"

# סף רעש: אם הפער בין ה-AI למרצה גדול מזה, נזרוק את הדוגמה
NOISE_THRESHOLD = 30 

# ==========================================
# 1. טעינת הנתונים וסינון רעשים חכם
# ==========================================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found. Run extract_features.py first.")
    exit()

print("📊 Loading dataset...")
raw_df = pd.read_csv(INPUT_FILE)
print(f"   Original dataset size: {len(raw_df)}")

clean_rows = []

print("\n🧹 Running Noise Filter (Sanity Check)...")
print("-" * 60)
for index, row in raw_df.iterrows():
    # חישוב "ציון משוער" גס של ה-AI (ממוצע של הפרמטרים)
    # זה לא הציון הסופי, אלא רק אינדיקציה לבדיקת שפיות
    ai_rough_score = (row['logic'] + row['accuracy'] + row['clarity']) / 3
    
    real_grade = row['real_grade']
    
    # חישוב הפער (הדיסוננס)
    gap = abs(ai_rough_score - real_grade)
    
    # הסינון: אם הפער ענק, זה רעש
    if gap < NOISE_THRESHOLD:
        clean_rows.append(row)
        status = "✅ Keep"
    else:
        status = "🗑️ DROP (Noise)"
    
    print(f"File: {row['filename'][:15]:<15} | AI View: {ai_rough_score:.0f} | Real: {real_grade} | Gap: {gap:.0f} | {status}")

print("-" * 60)

# יצירת הדאטה-בייס הנקי
df = pd.DataFrame(clean_rows)
print(f"📉 Final dataset size: {len(df)} (Dropped {len(raw_df) - len(df)} outliers)\n")

# הכנת הנתונים לאימון
X = df[['logic', 'accuracy', 'clarity']].values
y = df['real_grade'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_FILE)

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# ==========================================
# 2. המודל הליניארי (היציב)
# ==========================================
class GraderNet(nn.Module):
    def __init__(self):
        super(GraderNet, self).__init__()
        # מודל ליניארי פשוט: שכבה אחת
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

model = GraderNet()

# ==========================================
# 3. אימון (Training)
# ==========================================
# הגדרות אגרסיביות כדי להגיע לתוצאה מהר ומדויק
optimizer = optim.Adam(model.parameters(), lr=0.1) 
criterion = nn.MSELoss()
epochs = 5000 

print("🚀 Starting Training on Clean Data...")

for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X_tensor)
    loss = criterion(prediction, y_tensor)
    loss.backward()
    optimizer.step()

print("✅ Training Complete!")

# ==========================================
# 4. שמירה ובדיקה
# ==========================================
torch.save(model.state_dict(), MODEL_FILE)
print(f"💾 Model saved to: {MODEL_FILE}")
print(f"💾 Scaler saved to: {SCALER_FILE}")

# הדפסת המשקולות (כדי לוודא שהמודל למד ש-Accuracy זה חשוב)
weights = model.fc1.weight.data.numpy()[0]
print("\n🧠 Learned Weights (Importance):")
print(f"   Logic:    {weights[0]:.2f}")
print(f"   Accuracy: {weights[1]:.2f}")
print(f"   Clarity:  {weights[2]:.2f}")

print("\n🔍 Final Predictions (On Clean Data):")
with torch.no_grad():
    test_pred = model(X_tensor).numpy()
    for i in range(len(df)):
        fname = df.iloc[i]['filename']
        real = df.iloc[i]['real_grade']
        pred = min(100, max(0, test_pred[i][0]))
        print(f"{fname:<20} | Real: {real:<3} | Pred: {pred:.1f}")