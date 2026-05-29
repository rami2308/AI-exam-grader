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
# Settings
# ==========================================
INPUT_FILE = "training_features.csv"
MODEL_FILE = "grader_model.pth"
SCALER_FILE = "scaler.pkl"
GRAPH_FILE = "loss_curve.png"

# Noise threshold: if the gap between the AI and the lecturer is larger than this, drop the sample
NOISE_THRESHOLD = 30

# ==========================================
# 1. Loading Data and Smart Noise Filtering
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
    # Calculate a rough "estimated score" from the AI (average of parameters)
    # This is not the final score, just an indicator for sanity checking
    ai_rough_score = row['coverage']

    real_grade = row['real_grade']

    # Calculate the gap (the dissonance)
    gap = abs(ai_rough_score - real_grade)

    # Filter: if the gap is huge, it's noise
    if gap < NOISE_THRESHOLD:
        clean_rows.append(row)
        status = "✅ Keep"
    else:
        status = "🗑️ DROP (Noise)"

    print(f"File: {row['filename'][:15]:<15} | AI View: {ai_rough_score:.0f} | Real: {real_grade} | Gap: {gap:.0f} | {status}")

print("-" * 60)

# Create the clean dataset
df = pd.DataFrame(clean_rows)
print(f"📉 Final dataset size: {len(df)} (Dropped {len(raw_df) - len(df)} outliers)\n")

# Prepare data for training
X = df[['coverage', 'mistakes']].values
y = df['real_grade'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_FILE)

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# ==========================================
# 2. The Linear Model (Stable)
# ==========================================
class GraderNet(nn.Module):
    def __init__(self):
        super(GraderNet, self).__init__()
        # Simple linear model: one layer
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

model = GraderNet()

# ==========================================
# 3. Training
# ==========================================
# Aggressive settings to reach results quickly and accurately
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
# 4. Saving and Verification
# ==========================================
torch.save(model.state_dict(), MODEL_FILE)
print(f"💾 Model saved to: {MODEL_FILE}")
print(f"💾 Scaler saved to: {SCALER_FILE}")

# Print weights to verify the model learned that Accuracy is important
weights = model.fc1.weight.data.numpy()[0]
print("\n🧠 Learned Weights (Importance):")
print(f"   Coverage: {weights[0]:.2f}")
print(f"   Mistakes: {weights[1]:.2f}")

print("\n🔍 Final Predictions (On Clean Data):")
with torch.no_grad():
    test_pred = model(X_tensor).numpy()
    for i in range(len(df)):
        fname = df.iloc[i]['filename']
        real = df.iloc[i]['real_grade']
        pred = min(100, max(0, test_pred[i][0]))
        print(f"{fname:<20} | Real: {real:<3} | Pred: {pred:.1f}")
