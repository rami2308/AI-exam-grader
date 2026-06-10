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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE.upper()}")

if DEVICE == "cuda":
    torch.cuda.empty_cache()

print("Loading Qwen Vision Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
) if DEVICE == "cuda" else None

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
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
You are a strict academic grader evaluating a student's exam answer.

Image 1: The exam question
Image 2: The professor's CORRECT answer (the answer key)
Image 3: The student's answer

STEP 1 — Decompose the correct answer into weighted components:
Assign each component a weight (weights must sum to 100). Distinguish between:
- Core concepts (30-50 pts each): the essential ideas the question is testing
- Supporting points (15-25 pts each): important details
- Minor details (5-10 pts): small additions that show depth

STEP 2 — Score the student per component (partial credit allowed):
- 100%: fully correct, clearly demonstrated understanding
- 50-75%: partially correct or incomplete but on the right track
- 25%: mentioned the concept but with significant errors or vagueness
- 0%: missing, wrong, or contradicts the correct answer

STEP 3 — Deduct for errors:
For each piece of incorrect or misleading information the student wrote (not just omissions):
Minor error: -5 pts. Major error: -10 to -20 pts.

STEP 4 — Calculate:
Multiply each component's weight by the student's percentage for that component. Sum them. Apply deductions. Clamp to [0, 100].

Write one sentence:
Summary: [what the student got right, partially right, and wrong]

Then:
Coverage: [0-100]
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
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    clean_response = full_output.split("assistant")[-1] if "assistant" in full_output else full_output

    try:
        coverage = int(re.search(r"Coverage\D*(\d+)", clean_response, re.IGNORECASE).group(1))

        summary_match = re.search(r"Summary:\s*(.+)", clean_response, re.IGNORECASE)
        explanation = summary_match.group(1).strip() if summary_match else "No explanation generated"

        print(f"[{index+1}/{len(df)}] {filename}: Coverage={coverage} (Real: {real_grade})")
        print(f"   → {explanation}")

        processed_data.append({
            "filename":    filename,
            "coverage":    coverage,
            "explanation": explanation,
            "real_grade":  real_grade
        })
    except Exception as e:
        print(f"Parsing failed for {filename}. Response was:\n{clean_response}\n")

if processed_data:
    pd.DataFrame(processed_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")