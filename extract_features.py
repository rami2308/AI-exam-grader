import os
import pandas as pd
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# הגדרות
# ==========================================
INPUT_CSV = "train.csv"
OUTPUT_CSV = "training_features.csv"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# ניקוי זיכרון
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==========================================
# טעינת המודל (יציב וחסכוני)
# ==========================================
print("👁️  Loading Qwen Vision Model (Fair Grader Mode)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==========================================
# הלוגיקה
# ==========================================
if not os.path.exists(INPUT_CSV):
    print("❌ train.csv not found! Run generate script first.")
    exit()

df = pd.read_csv(INPUT_CSV)
processed_data = []

# הפרומפט המדויק
EXTRACT_PROMPT = """
Act as a fair Professor. Analyze the student's solution.
Provide scores based on Logic and Correctness (0-100).

GUIDELINES:
- Perfect answer = 100.
- Correct Logic but minor Syntax error (missing ;, typos) = Score 70-85.
- Correct Logic but wrong output format = Score 60-75.
- Logic wrong = Score 0-50.

Format strictly:
Logic: [Score]
Accuracy: [Score]
Clarity: [Score]
"""

print(f"🔄 Processing {len(df)} exams...")

for index, row in df.iterrows():
    filename = row['filename']
    real_grade = row['grade']
    
    q_path = os.path.join("data/questions", filename)
    s_path = os.path.join("data/solutions", filename)

    if not os.path.exists(q_path) or not os.path.exists(s_path):
        continue

    messages = [{"role": "user", "content": [
        {"type": "image", "image": q_path},
        {"type": "image", "image": s_path},
        {"type": "text", "text": EXTRACT_PROMPT}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.2)
    
    full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    clean_response = full_output.split("assistant")[-1] if "assistant" in full_output else full_output

    try:
        # Regex חזק
        logic = int(re.search(r"Logic\D*(\d+)", clean_response, re.IGNORECASE).group(1))
        acc = int(re.search(r"Accuracy\D*(\d+)", clean_response, re.IGNORECASE).group(1))
        clarity = int(re.search(r"Clarity\D*(\d+)", clean_response, re.IGNORECASE).group(1))
        
        # תיקון סקאלה אוטומטי
        if logic <= 10: logic *= 10
        if acc <= 10: acc *= 10
        if clarity <= 10: clarity *= 10

        print(f"✅ [{index+1}/{len(df)}] {filename}: L={logic:<3} A={acc:<3} C={clarity:<3} (Real: {real_grade})")
        
        processed_data.append({
            "filename": filename, 
            "logic": logic, 
            "accuracy": acc, 
            "clarity": clarity, 
            "real_grade": real_grade
        })
    except:
        print(f"❌ Parsing failed for {filename}.")

if processed_data:
    pd.DataFrame(processed_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 Data Extraction Complete! Saved to '{OUTPUT_CSV}'")