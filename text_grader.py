import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# Question Settings (lecturer enters data here)
# ==========================================
# Example: History/Civics question
QUESTION = "Explain the principle of 'Separation of Powers' in a democracy."
REFERENCE_ANSWER = """
The principle of separation of powers divides the government into three branches:
1. Legislative (makes laws).
2. Executive (enforces laws).
3. Judicial (interprets laws).
The goal is to prevent concentration of power and enable checks and balances.
"""

# ==========================================
# Model Settings (same powerful engine)
# ==========================================
EXERCISE_FOLDER = "data/text_answers"  # Note the new folder
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"Loading AI Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model Loaded! 🚀")

# ==========================================
# Prompt adapted for open-ended questions
# ==========================================
SYSTEM_PROMPT = f"""
You are a History/Civics Professor grading a handwritten exam.
Compare the student's handwritten answer to the REFERENCE ANSWER provided below.

### QUESTION:
{QUESTION}

### REFERENCE ANSWER (Correct Solution):
{REFERENCE_ANSWER}

### GRADING INSTRUCTIONS:
1. **Transcribe** the student's handwriting exactly as is.
2. **Analyze Content:** Check if the student mentioned the key concepts from the reference answer.
   - Did they mention the 3 branches?
   - Did they explain the goal (checks and balances)?
   - Do NOT penalize for spelling mistakes unless the meaning is lost.
   - Accept synonyms (e.g., "Parliament" instead of "Legislative" is okay).

### OUTPUT FORMAT:
## 1. Transcription
(Student's text)

## 2. Key Point Check
* Legislative/Making laws: [V/X]
* Executive/Enforcing laws: [V/X]
* Judicial/Interpreting laws: [V/X]
* Checks and Balances concept: [V/X]

## 3. Final Score
Score: [0-100]
Feedback: (One sentence summary).
"""

# ==========================================
# Grading Loop
# ==========================================
while True:
    print("\n" + "="*40)
    print("📂 Available Text Answers:")

    if not os.path.exists(EXERCISE_FOLDER):
        os.makedirs(EXERCISE_FOLDER)

    files = [f for f in os.listdir(EXERCISE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()

    if not files:
        print(f"❌ No images found in {EXERCISE_FOLDER}")
        print("Please upload handwritten text images.")
        break

    for i, file_name in enumerate(files):
        print(f" [{i+1}] {file_name}")
    print(" [q] Quit")

    choice = input("\nSelect an answer to grade (1-9) or 'q': ")

    if choice.lower() == 'q':
        break

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            selected_image = files[idx]
            image_path = os.path.join(EXERCISE_FOLDER, selected_image)

            print(f"\n📝 Reading Handwriting: {selected_image}...")

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

            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print("\n" + "-"*15 + " GRADING REPORT " + "-"*15)
            print(output_text[0])
            print("-" * 46)
            input("\nPress Enter to continue...")

        else:
            print("❌ Invalid number.")
    except ValueError:
        print("❌ Invalid input.")
