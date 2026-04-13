import torch
import torch.nn as nn
import re, os, joblib
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# הגדרות
# ==========================================
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
GRADER_MODEL = "grader_model.pth"
SCALER_FILE = "scaler.pkl"
QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"

# ==========================================
# 1. המוח (Linear Grader)
# ==========================================
class GraderNet(nn.Module):
    def __init__(self):
        super(GraderNet, self).__init__()
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc1(x)

# ==========================================
# 2. טעינת המערכת
# ==========================================
print("\n" + "="*50)
print("⚙️  INITIALIZING FINAL SYSTEM...")
print("="*50)

print("1. Loading Vision Model (Qwen)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("2. Loading Trained Grader Network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brain = GraderNet().to(device)
brain.load_state_dict(torch.load(GRADER_MODEL, map_location=device))
brain.eval()
scaler = joblib.load(SCALER_FILE)
print("✅ System Ready! Waiting for user input...\n")

# ==========================================
# 3. פונקציית הבדיקה (המתוקנת!)
# ==========================================
def grade_specific_exam(filename):
    q_path = os.path.join(QUESTIONS_DIR, filename)
    s_path = os.path.join(SOLUTIONS_DIR, filename)
    if not os.path.exists(q_path): q_path = os.path.join("data/questions", filename)

    print(f"\n📄 Analyzing: {filename}...")
    
    # === הפרומפט החדש וההוגן ===
    prompt = """
    You are a fair and helpful Teaching Assistant.
    Analyze the student's solution compared to the question.
    
    SCORING GUIDELINES (0-100 Scale):
    1. LOGIC IS MOST IMPORTANT. If the logic is correct but syntax is slightly wrong, give a HIGH score (70-85).
    2. Only give 0-40 if the code makes no sense at all.
    3. If they used () instead of [] but the logic works, deduct only small points.
    
    Format:
    Logic: [Score 0-100]
    Accuracy: [Score 0-100]
    Clarity: [Score 0-100]
    Feedback: [Short constructive feedback]
    """
    
    messages = [{"role": "user", "content": [{"type": "image", "image": q_path}, {"type": "image", "image": s_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Temperature 0.4 נותן לו קצת יותר גמישות "לחשוב" ולא להינעל על חוקים יבשים
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.4) 
    
    output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    clean_output = output.split("assistant")[-1] if "assistant" in output else output

    try:
        # Regex שתופס מספרים ומתקן אוטומטית אם המודל נתן ציון חד ספרתי (למשל 8 הופך ל-80)
        def extract_score(pattern, text):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                return val * 10 if val <= 10 else val # תיקון אוטומטי
            return 0

        logic = extract_score(r"Logic\D*(\d+)", clean_output)
        acc = extract_score(r"Accuracy\D*(\d+)", clean_output)
        clarity = extract_score(r"Clarity\D*(\d+)", clean_output)
        
        feedback = "Evaluated."
        if "Feedback" in clean_output:
            feedback = clean_output.split("Feedback")[-1].replace(":", "").strip().split("\n")[0]

        # חישוב סופי
        pred = brain(torch.FloatTensor(scaler.transform([[logic, acc, clarity]])).to(device)).item()
        final_grade = min(100, max(0, int(pred)))

        print("\n" + "╔" + "═"*50 + "╗")
        print(f"║ 🎓 FINAL GRADE: {final_grade:>3}/100                     ║")
        print("╠" + "═"*50 + "╣")
        print(f"║ 📊 Logic:    {logic:<3}                                 ║")
        print(f"║ 📊 Accuracy: {acc:<3}                                 ║")
        print(f"║ 📊 Clarity:  {clarity:<3}                                 ║")
        print("╠" + "═"*50 + "╣")
        print(f"║ 📝 Feedback:                                      ║")
        print(f"║ {feedback[:46]:<46} ║") 
        print("╚" + "═"*50 + "╝\n")

    except Exception as e:
        print(f"❌ Error: {e}\nRaw Output: {clean_output}")

# ==========================================
# 4. Loop
# ==========================================
while True:
    files = sorted([f for f in os.listdir(SOLUTIONS_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    if not files: break
    print("\n📚 Available Exams:"); 
    for i, f in enumerate(files): print(f" [{i+1}] {f}")
    choice = input("👉 Select exam number: ")
    if choice == 'q': break
    try: grade_specific_exam(files[int(choice)-1])
    except: pass