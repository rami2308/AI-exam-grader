from PIL import Image as PILImage

def resize_for_vlm(path, out, max_w=800, max_h=600):
    img = PILImage.open(path)
    img.thumbnail((max_w, max_h))
    img.save(out)
    return out


"""
AI Exam Grader — Demo Script
-----------------------------
Usage:
  1. Place your 3 images with the SAME filename in:
       data/questions/myexam.png
       data/solutions/myexam.png
       data/answers/myexam.png

  2. Run:
       python demo.py

  3. Pick the exam from the menu and see the predicted grade.
"""

import os, re, torch, torch.nn as nn, joblib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "Qwen/Qwen2.5-VL-7B-Instruct"
GRADER_MODEL  = "grader_model.pth"
SCALER_FILE   = "scaler.pkl"
QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"
ANSWERS_DIR   = "data/answers"

# ── GraderNet (must match train_model.py exactly) ─────────────────────────
class GraderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(1, 16)
        self.fc2  = nn.Linear(16, 8)
        self.fc3  = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ── Load everything once at startup ───────────────────────────────────────
print("\n" + "="*55)
print("  AI EXAM GRADER — Loading System...")
print("="*55)

print("  [1/3] Loading Vision Model (Qwen2.5-VL-7B)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
vlm       = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("  [2/3] Loading Trained GraderNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brain  = GraderNet().to(device)
brain.load_state_dict(torch.load(GRADER_MODEL, map_location=device))
brain.eval()
scaler = joblib.load(SCALER_FILE)

print("  [3/3] Ready.\n")

# ── Coverage prompt (same as extract_features.py / training) ──────────────
COVERAGE_PROMPT = """
You are a strict academic grader.
Image 1 is the exam question.
Image 2 is the CORRECT answer provided by the professor.
Image 3 is the STUDENT's answer.

Follow these steps out loud:
1. List the key concepts or steps from the CORRECT answer.
2. Check which of those concepts the student covered. Grade by meaning, not exact wording.
3. Count: how many concepts did the student get right out of total?
4. Note any wrong or misleading information the student added.

Before your final answer, write one summary sentence in this exact format:
Summary: [how many key concepts were in the correct answer, how many the student covered, and what was missing if anything]

Then write:
Coverage: [0-100]  (this is a PERCENTAGE, not a count. If student covered 2 out of 3 concepts, Coverage = 67, not 2)
"""

# ── Grade one exam ─────────────────────────────────────────────────────────
def grade(filename):
    q = os.path.join(QUESTIONS_DIR, filename)
    s = os.path.join(SOLUTIONS_DIR, filename)
    a = os.path.join(ANSWERS_DIR,   filename)

    for path, label in [(q, "Question"), (s, "Solution"), (a, "Answer")]:
        if not os.path.exists(path):
            print(f"  ❌  Missing file: {path}  ({label})")
            return
    
    q = resize_for_vlm(q, "/tmp/tmp_q.jpg")
    s = resize_for_vlm(s, "/tmp/tmp_s.jpg")
    a = resize_for_vlm(a, "/tmp/tmp_a.jpg")
    print(f"\n  Analyzing: {filename}")
    print("  Running VLM... (this takes ~20-40 seconds)")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": q},
        {"type": "image", "image": a},
        {"type": "image", "image": s},
        {"type": "text",  "text": COVERAGE_PROMPT}
    ]}]

    text         = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs       = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_ids = vlm.generate(**inputs, max_new_tokens=1024, do_sample=False)

    raw     = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    output  = raw.split("assistant")[-1] if "assistant" in raw else raw

    # Extract Coverage score
    match = re.search(r"Coverage\D*(\d+)", output, re.IGNORECASE)
    if not match:
        print("  ❌  Could not parse Coverage from VLM output.")
        print(f"  Raw output:\n{output}")
        return
    print("\n--- RAW VLM OUTPUT ---")
    print(output)
    print("--- END ---\n")

    coverage = int(match.group(1))

    # Extract summary
    summary_match = re.search(r"Summary:\s*(.+)", output, re.IGNORECASE)
    summary = summary_match.group(1).strip() if summary_match else "(no summary)"

    # Run GraderNet
    scaled = scaler.transform([[coverage]])
    with torch.no_grad():
        pred = brain(torch.FloatTensor(scaled).to(device)).item()
    final_grade = int(min(100, max(0, round(pred))))

    # Print result
    print("\n" + "╔" + "═"*51 + "╗")
    print(f"║  🎓  PREDICTED GRADE:  {final_grade:>3} / 100               ║")
    print("╠" + "═"*51 + "╣")
    print(f"║  Coverage Score (VLM):  {coverage:<3}                      ║")
    print("╠" + "═"*51 + "╣")
    summary_line = summary[:47]
    print(f"║  Summary: {summary_line:<47} ║")
    print("╚" + "═"*51 + "╝\n")

# ── Main menu ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(ANSWERS_DIR, exist_ok=True)

    while True:
        # Show only exams that have all 3 matching images
        all_files = sorted(f for f in os.listdir(ANSWERS_DIR)
                           if f.lower().endswith((".png", ".jpg", ".jpeg")))
        ready = [f for f in all_files
                 if os.path.exists(os.path.join(QUESTIONS_DIR, f))
                 and os.path.exists(os.path.join(SOLUTIONS_DIR, f))]

        if not ready:
            print("\n  No exams ready to grade.")
            print(f"  Add matching images to:")
            print(f"    {QUESTIONS_DIR}/")
            print(f"    {SOLUTIONS_DIR}/")
            print(f"    {ANSWERS_DIR}/")
            print("  Then re-run.\n")
            break

        print("\n  Available exams:")
        for i, f in enumerate(ready):
            print(f"    [{i+1}] {f}")

        choice = input("\n  Select exam number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            break
        try:
            grade(ready[int(choice) - 1])
        except (ValueError, IndexError):
            print("  Invalid choice.")

if __name__ == "__main__":
    main()
