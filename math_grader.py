import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# הגדרות מערכת
# ==========================================
EXERCISE_FOLDER = "data/exercises"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# ==========================================
# 1. טעינת המודל (4-bit למניעת קריסה)
# ==========================================
print(f"Loading AI Model from {MODEL_PATH}...")
print("Configuring 4-bit quantization for Tesla T4 compatibility...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model Loaded Successfully! 🚀")

# ==========================================
# 2. הלוגיקה החדשה (ECF Support)
# ==========================================
SYSTEM_PROMPT = """
You are an empathetic but precise Math Professor grading an exam based on a strict rubric.
Your goal is to evaluate the student's UNDERSTANDING, not just the final number.

## GRADING RUBRIC (Total: 100 points):
1. **Method & Logic (70 points):** Did the student choose the correct formula/approach? Are the steps logically connected?
   * CRITICAL: Apply "Error Carried Forward" (ECF). If the student makes a calculation error (e.g., 2+3=6) but continues the rest of the steps correctly using that wrong number, DO NOT penalize the subsequent steps. Only penalize the specific calculation error.
2. **Calculation Accuracy (20 points):** Deduct points here for arithmetic mistakes (sign errors, wrong addition/multiplication).
3. **Final Answer (10 points):** Is the final result correct based on the original problem?

## OUTPUT FORMAT:
(Do not write LaTeX preambles like \\documentclass).

### 1. Solution Transcription
(Write the student's solution in clean LaTeX math format).

### 2. Step-by-Step Analysis
* **Step 1:** [Correct/Error] - Explanation.
* **Step 2:** [Correct/Error/ECF] - Explanation. (Note if it's a Follow-Through error).
...

### 3. Grading Breakdown
* **Method & Logic:** X/70
* **Calculation Accuracy:** Y/20
* **Final Answer:** Z/10

### 4. Final Score & Feedback
**Total Score: [SUM]/100**
**Feedback:** (Summarize the main mistake if any, e.g., "Great logic, but you made a sign error in line 2 which affected the final result").
"""

# ==========================================
# 3. לולאת הבחירה והבדיקה
# ==========================================
while True:
    print("\n" + "="*40)
    print("📂 Available Exercises:")
    
    if not os.path.exists(EXERCISE_FOLDER):
        os.makedirs(EXERCISE_FOLDER)
        
    files = [f for f in os.listdir(EXERCISE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()

    if not files:
        print(f"❌ No images found in {EXERCISE_FOLDER}")
        print("Please upload images and try again.")
        break

    for i, file_name in enumerate(files):
        print(f" [{i+1}] {file_name}")
    print(" [q] Quit")

    choice = input("\nSelect an image to grade (1-9) or 'q': ")

    if choice.lower() == 'q':
        print("Exiting...")
        break

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            selected_image = files[idx]
            image_path = os.path.join(EXERCISE_FOLDER, selected_image)
            
            print(f"\n📝 Grading: {selected_image} with Logic-First Rubric...")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print("\n" + "-"*15 + " TEACHER REPORT " + "-"*15)
            print(output_text[0])
            print("-" * 46)
            input("\nPress Enter to continue...") 

        else:
            print("❌ Invalid number.")
    except ValueError:
        print("❌ Invalid input.")