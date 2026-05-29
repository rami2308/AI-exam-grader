import os
import pandas as pd
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# Settings
# ==========================================
INPUT_CSV     = "train.csv"
OUTPUT_CSV    = "training_features.csv"
MODEL_PATH    = "Qwen/Qwen2.5-VL-7B-Instruct"
QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"
ANSWERS_DIR   = "data/answers"

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Loading Qwen Vision Model...")
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
    print(f"Error loading model: {e}")
    exit()

if not os.path.exists(INPUT_CSV):
    print("train.csv not found. Run generate_synthetic_data.py first.")
    exit()

df = pd.read_csv(INPUT_CSV)
processed_data = []

EXTRACT_PROMPT = """
You are a strict academic grader.
Image 1 is the exam question.
Image 2 is the CORRECT answer provided by the professor.
Image 3 is the STUDENT's answer.

Your job:
1. Read the correct answer and identify its key concepts or steps.
2. Check how many of those key concepts the student actually covered.
3. The student may use different words or phrasing than the correct answer.
   Grade based on meaning and concepts, not exact wording.

Respond with ONLY these two lines, nothing else:
Coverage: [0-100]
Mistakes: [0-100]

Coverage = percentage of the correct answer's key concepts the student captured.
Mistakes = how much wrong or misleading information the student added (0 = nothing wrong, 100 = completely wrong).
"""

print(f"Processing {len(df)} exams...")

for index, row in df.iterrows():
    filename  = row['filename']
    real_grade = row['grade']

    q_path = os.path.join(QUESTIONS_DIR, filename)
    s_path = os.path.join(SOLUTIONS_DIR, filename)
    a_path = os.path.join(ANSWERS_DIR,   filename)

    if not os.path.exists(q_path) or not os.path.exists(s_path) or not os.path.exists(a_path):
        print(f"Skipping {filename} - missing image file.")
        continue

    messages = [{"role": "user", "content": [
        {"type": "image", "image": q_path},
        {"type": "image", "image": a_path},
        {"type": "image", "image": s_path},
        {"type": "text",  "text": EXTRACT_PROMPT}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    clean_response = full_output.split("assistant")[-1] if "assistant" in full_output else full_output

    try:
        coverage = int(re.search(r"Coverage\D*(\d+)", clean_response, re.IGNORECASE).group(1))
        mistakes = int(re.search(r"Mistakes\D*(\d+)", clean_response, re.IGNORECASE).group(1))

        print(f"[{index+1}/{len(df)}] {filename}: Coverage={coverage} Mistakes={mistakes} (Real: {real_grade})")

        processed_data.append({
            "filename":   filename,
            "coverage":   coverage,
            "mistakes":   mistakes,
            "real_grade": real_grade
        })
    except Exception as e:
        print(f"Parsing failed for {filename}. Response was:\n{clean_response}\n")

if processed_data:
    pd.DataFrame(processed_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")