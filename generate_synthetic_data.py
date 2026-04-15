import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# Settings
# ==========================================
QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"
CSV_FILE = "train.csv"

os.makedirs(QUESTIONS_DIR, exist_ok=True)
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# ==========================================
# The Full and Extended Dataset (The Mega Dataset)
# ==========================================
dataset = [
    # --- Group 1: Perfect (100) ---
    {"f": "algo_q1b_100.png", "q": "Calculate T(n) for Merge Sort.", "s": "T(n) = 2T(n/2) + O(n). Final: O(n log n).", "g": 100},
    {"f": "java_loop_100.png", "q": "Write a for loop 1-5.", "s": "for(int i=1; i<=5; i++) System.out.println(i);", "g": 100},
    {"f": "sql_join_100.png", "q": "Join users and orders.", "s": "SELECT * FROM users JOIN orders ON u.id=o.uid;", "g": 100},
    {"f": "py_fibo_100.png", "q": "Fibonacci recursive.", "s": "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)", "g": 100},
    {"f": "algo_hash_100.png", "q": "Hash map access time?", "s": "O(1) on average.", "g": 100},
    {"f": "os_thread_100.png", "q": "Thread vs Process?", "s": "Threads share memory, processes do not.", "g": 100},
    {"f": "py_list_comp_100.png", "q": "List of squares 0-9.", "s": "[x**2 for x in range(10)]", "g": 100},

    # --- Group 2: Failing (0-40) ---
    {"f": "algo_q2_0.png", "q": "Explain QuickSort worst case.", "s": "It is O(n) because it is fast.", "g": 0},
    {"f": "sql_delete_0.png", "q": "Delete user id 5.", "s": "DELETE FROM users;", "g": 0},  # Deleted everything!
    {"f": "java_syntax_20.png", "q": "Print Hello.", "s": "print 'Hello'", "g": 20},  # Not Java at all
    {"f": "algo_binary_40.png", "q": "Binary search req?", "s": "Array must be numbers.", "g": 40},  # Forgot that sorting is required

    # --- Group 3: Almost Very Good (80-90) ---
    {"f": "java_class_88.png", "q": "Define class Car.", "s": "class car { } // Should be capital C", "g": 88},
    {"f": "sql_semi_90.png", "q": "Update status.", "s": "UPDATE users SET status='active' // No semicolon", "g": 90},
    {"f": "py_colon_90.png", "q": "Func add(a,b).", "s": "def add(a,b)\n return a+b // Missing colon", "g": 90},
    {"f": "algo_dfs_85.png", "q": "Explain DFS.", "s": "DFS uses a Stack. It goes deep first.", "g": 85},  # Correct but too short
    {"f": "java_print_85.png", "q": "Print 'Hi'.", "s": "System.out.print('Hi') // Missing ;", "g": 85},

    # --- Group 4: OK but has mistakes (70-79) ---
    {"f": "py_tuple_78.png", "q": "Create list 1-3.", "s": "x = (1, 2, 3) // Created tuple not list", "g": 78},
    {"f": "java_types_75.png", "q": "Int variable x=10.", "s": "x = 10; // Missing 'int'", "g": 75},
    {"f": "java_brackets_72.png", "q": "If x > 5.", "s": "if x > 5 { } // Missing ()", "g": 72},
    {"f": "algo_recur_75.png", "q": "Recursion base case?", "s": "To stop the loop.", "g": 75},  # Too informal an explanation
    {"f": "sql_quotes_75.png", "q": "Select name 'Dan'.", "s": "SELECT * FROM u WHERE name=\"Dan\"", "g": 75},  # Double quotes not standard in SQL

    # --- Group 5: Barely passing / average (60-69) ---
    {"f": "java_loop_60.png", "q": "Loop 5 times.", "s": "while(i<5) print(i); // Infinite loop", "g": 60},
    {"f": "py_indent_65.png", "q": "If x=5 print yes.", "s": "if x==5:\nprint('yes') // Indentation error", "g": 65},
    {"f": "sql_join_65.png", "q": "Join orders.", "s": "SELECT * FROM orders JOIN users; // No ON clause", "g": 65},
    {"f": "py_import_60.png", "q": "Import math.", "s": "include math", "g": 60},  # C syntax
    {"f": "gen_api_60.png", "q": "What is API?", "s": "Application Interface.", "g": 60},  # Forgot 'Programming'

    # --- Group 6: Confusion tests (subtle logical errors) ---
    {"f": "py_range_68.png", "q": "Loop 0 to 4.", "s": "for i in range(5)\n print i", "g": 68},  # Missing colon and old print syntax
    {"f": "java_ret_68.png", "q": "Return 5.", "s": "void get() { return 5; }", "g": 68},  # Void cannot return a value
    {"f": "sql_form_55.png", "q": "Select all.", "s": "SELECT * FORM users", "g": 55},  # Typo: FORM instead of FROM
    {"f": "algo_stack_50.png", "q": "Stack FIFO?", "s": "Yes, Stack is FIFO.", "g": 50},  # Factual error: Stack is LIFO
    {"f": "py_var_70.png", "q": "Set valid=True.", "s": "valid-user = True", "g": 70},  # Hyphen in variable name is invalid

    # --- Final additions for variety ---
    {"f": "os_deadlock_80.png", "q": "What is Deadlock?", "s": "When two processes wait for each other.", "g": 80},
    {"f": "net_tcp_95.png", "q": "TCP vs UDP", "s": "TCP reliable, UDP fast.", "g": 95},
    {"f": "cpp_cout_85.png", "q": "Print in C++", "s": "cout << 'Hello'", "g": 85}  # Missing std:: or semicolon
]

def create_image_from_text(text, filename, folder):
    W, H = 800, 600
    img = Image.new('RGB', (W, H), color='white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()

    # Blue lines for a notebook page appearance
    for i in range(50, H, 50):
        d.line([(0, i), (W, i)], fill=(200, 220, 255), width=2)
    d.line([(60, 0), (60, H)], fill=(255, 200, 200), width=2)

    lines = text.split('\n')
    y_text = 65
    for line in lines:
        d.text((70, y_text), line, fill=(0, 0, 0), font=font)
        y_text += 35

    path = os.path.join(folder, filename)
    img.save(path)
    print(f"🖼️  Created: {filename} -> Grade: {next(i['g'] for i in dataset if i['f'] == filename)}")

# ==========================================
# Main
# ==========================================
print("🏭 Generating MEGA Dataset (40+ Exams)...")

# Delete old file for a clean build
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

new_rows = []
for item in dataset:
    create_image_from_text(item['q'], item['f'], QUESTIONS_DIR)
    create_image_from_text(item['s'], item['f'], SOLUTIONS_DIR)
    new_rows.append({'filename': item['f'], 'grade': item['g']})

pd.DataFrame(new_rows).to_csv(CSV_FILE, index=False)
print(f"\n✅ Successfully created {len(new_rows)} exams in train.csv!")
