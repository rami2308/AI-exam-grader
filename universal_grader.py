import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. Path Settings
# ==========================================
QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# ==========================================
# 2. Loading the Model
# ==========================================
print(f"Loading Universal Grader Brain...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("System Ready! 🚀")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# 3. The Refined Prompt (Chain of Thought)
# ==========================================
UNIVERSAL_PROMPT = """
You are a Precise and Fair Exam Grader.
I will show you two images:
1. **THE QUESTION** (Printed).
2. **THE STUDENT'S SOLUTION** (Handwriting).

YOUR MISSION:
1. **Analyze the Question:** Determine the subject and solve it mentally first.
2. **Analyze the Student's Work:** Read their steps and final answer.
3. **Compare & Grade:**

   - **FOR MATH/PHYSICS (CRITICAL):**
     * **Step 1:** Verify the formulas used.
     * **Step 2:** Verify the calculation logic.
     * **Step 3:** Compare the student's Final Answer to YOUR calculated answer.
     * **RULE:** If the student's number matches your number (e.g., 37.5 matches 37.5), mark it as CORRECT. Do not invent errors.
     * **ECF:** If a minor error occurred earlier, deduct points only once.

   - **FOR HUMANITIES:**
     * Check for key concepts and understanding.

OUTPUT FORMAT:
## 1. Subject Detected
(e.g., "Physics - Kinematics")

## 2. Step-by-Step Verification
* **Formula:** [Correct/Incorrect]
* **Substitution:** [Correct/Incorrect]
* **Calculation:** [Correct/Incorrect] - (Explicitly state: "Student wrote X, Correct is Y")

## 3. Final Score
Score: [0-100]
Feedback: (Constructive feedback).
"""

# ==========================================
# 4. The Main Engine
# ==========================================
while True:
    print("\n" + "="*50)
    print("🌍 UNIVERSAL GRADING SYSTEM (v2.0)")
    print("="*50)

    if not os.path.exists(SOLUTIONS_DIR) or not os.path.exists(QUESTIONS_DIR):
        print(f"❌ Error: Missing folders '{QUESTIONS_DIR}' or '{SOLUTIONS_DIR}'")
        break

    files = sorted([f for f in os.listdir(SOLUTIONS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    valid_pairs = []

    for f in files:
        q_path = os.path.join(QUESTIONS_DIR, f)
        if os.path.exists(q_path):
            valid_pairs.append(f)

    if not valid_pairs:
        print("❌ No matching pairs found.")
        input("Press Enter to retry...")
        continue

    for i, fname in enumerate(valid_pairs):
        print(f" [{i+1}] {fname}")

    choice = input("\nSelect pair to grade (or 'q' to quit): ")
    if choice.lower() == 'q': break

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(valid_pairs):
            filename = valid_pairs[idx]
            q_img_path = os.path.join(QUESTIONS_DIR, filename)
            s_img_path = os.path.join(SOLUTIONS_DIR, filename)

            print(f"\n📝 Analyzing: {filename}...")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "IMAGE 1: THE QUESTION"},
                        {"type": "image", "image": q_img_path},
                        {"type": "text", "text": "IMAGE 2: THE STUDENT'S SOLUTION"},
                        {"type": "image", "image": s_img_path},
                        {"type": "text", "text": UNIVERSAL_PROMPT},
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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print("\n" + "-"*20 + " GRADING REPORT " + "-"*20)
            print(output_text[0])
            print("-" * 56)
            input("\nPress Enter to check another exam...")

        else:
            print("Invalid number.")
    except ValueError:
        print("Invalid input.")
