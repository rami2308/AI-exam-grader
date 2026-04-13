from PIL import Image, ImageDraw, ImageFont
import os

# הגדרות
Q_DIR = "data/questions"
S_DIR = "data/solutions"
FILENAME = "blind_test_python.png"

# וודא שהתיקיות קיימות
os.makedirs(Q_DIR, exist_ok=True)
os.makedirs(S_DIR, exist_ok=True)

# התוכן: שאלה על סינון מספרים זוגיים
question_text = "Write a Python List Comprehension\nto create a list of even numbers from 0 to 10."

# התשובה של הסטודנט: נכונה לוגית, אבל יוצרת Generator () במקום List []
solution_text = "evens = (x for x in range(11) if x % 2 == 0)\n# Code works but creates a generator object, not a list."

def create_image(text, folder):
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    try:
        # ניסיון לטעון פונט, אחרת ברירת מחדל
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # קווים כחולים למראה של דף מחברת
    for i in range(50, 600, 50):
        d.line([(0, i), (800, i)], fill=(200, 220, 255), width=2)
    
    d.text((50, 60), text, fill=(0, 0, 0), font=font)
    path = os.path.join(folder, FILENAME)
    img.save(path)
    print(f"✅ Created: {path}")

# ביצוע
print("🎨 Generating blind test...")
create_image(question_text, Q_DIR)
create_image(solution_text, S_DIR)