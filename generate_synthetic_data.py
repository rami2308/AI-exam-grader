import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# Settings
# ==========================================
QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"
ANSWERS_DIR   = "data/answers"
CSV_FILE = "train.csv"

os.makedirs(QUESTIONS_DIR, exist_ok=True)
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(ANSWERS_DIR,   exist_ok=True)

# ==========================================
# Dataset
# q = question
# s = student answer
# a = professor's correct answer
# g = real grade
# ==========================================
dataset = [
    {
        "f": "algo_q1b_100.png",
        "q": "Calculate T(n) for Merge Sort.",
        "s": "T(n) = 2T(n/2) + O(n). Final: O(n log n).",
        "a": "T(n) = 2T(n/2) + O(n).\nBy Master Theorem: T(n) = O(n log n).",
        "g": 100
    },
    {
        "f": "java_loop_100.png",
        "q": "Write a for loop that prints 1 to 5.",
        "s": "for(int i=1; i<=5; i++) System.out.println(i);",
        "a": "for(int i=1; i<=5; i++) System.out.println(i);",
        "g": 100
    },
    {
        "f": "sql_join_100.png",
        "q": "Join users and orders tables.",
        "s": "SELECT * FROM users JOIN orders ON u.id=o.uid;",
        "a": "SELECT * FROM users u JOIN orders o ON u.id = o.uid;",
        "g": 100
    },
    {
        "f": "py_fibo_100.png",
        "q": "Write Fibonacci recursively in Python.",
        "s": "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
        "a": "def fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)",
        "g": 100
    },
    {
        "f": "algo_hash_100.png",
        "q": "What is the average access time of a hash map?",
        "s": "O(1) on average.",
        "a": "O(1) average case for lookup, insert, and delete.",
        "g": 100
    },
    {
        "f": "os_thread_100.png",
        "q": "What is the difference between a Thread and a Process?",
        "s": "Threads share memory, processes do not.",
        "a": "Threads share the same memory space and are lighter.\nProcesses have separate memory and are heavier.",
        "g": 100
    },
    {
        "f": "py_list_comp_100.png",
        "q": "Create a list of squares from 0 to 9 using list comprehension.",
        "s": "[x**2 for x in range(10)]",
        "a": "[x**2 for x in range(10)]",
        "g": 100
    },
    {
        "f": "algo_q2_0.png",
        "q": "Explain QuickSort worst case time complexity.",
        "s": "It is O(n) because it is fast.",
        "a": "O(n^2) - occurs when pivot is always the\nsmallest or largest element (e.g. sorted array).",
        "g": 0
    },
    {
        "f": "sql_delete_0.png",
        "q": "Delete the user with id = 5.",
        "s": "DELETE FROM users;",
        "a": "DELETE FROM users WHERE id = 5;",
        "g": 0
    },
    {
        "f": "java_syntax_20.png",
        "q": "Print the word Hello in Java.",
        "s": "print 'Hello'",
        "a": "System.out.println(\"Hello\");",
        "g": 20
    },
    {
        "f": "algo_binary_40.png",
        "q": "What is required for binary search to work?",
        "s": "Array must be numbers.",
        "a": "The array must be sorted in ascending order.",
        "g": 40
    },
    {
        "f": "java_class_88.png",
        "q": "Define a class named Car in Java.",
        "s": "class car { } // Should be capital C",
        "a": "class Car { }",
        "g": 88
    },
    {
        "f": "sql_semi_90.png",
        "q": "Update all users status to active.",
        "s": "UPDATE users SET status='active' // No semicolon",
        "a": "UPDATE users SET status='active';",
        "g": 90
    },
    {
        "f": "py_colon_90.png",
        "q": "Write a function add(a, b) that returns a+b.",
        "s": "def add(a,b)\n return a+b // Missing colon",
        "a": "def add(a, b):\n    return a + b",
        "g": 90
    },
    {
        "f": "algo_dfs_85.png",
        "q": "Explain how DFS (Depth First Search) works.",
        "s": "DFS uses a Stack. It goes deep first.",
        "a": "DFS uses a Stack (or recursion).\nIt explores as deep as possible before backtracking.",
        "g": 85
    },
    {
        "f": "java_print_85.png",
        "q": "Print the word Hi in Java.",
        "s": "System.out.print('Hi') // Missing ;",
        "a": "System.out.println(\"Hi\");",
        "g": 85
    },
    {
        "f": "py_tuple_78.png",
        "q": "Create a list containing 1, 2, 3 in Python.",
        "s": "x = (1, 2, 3) // Created tuple not list",
        "a": "x = [1, 2, 3]",
        "g": 78
    },
    {
        "f": "java_types_75.png",
        "q": "Declare an integer variable x with value 10.",
        "s": "x = 10; // Missing 'int'",
        "a": "int x = 10;",
        "g": 75
    },
    {
        "f": "java_brackets_72.png",
        "q": "Write an if statement that checks if x > 5.",
        "s": "if x > 5 { } // Missing ()",
        "a": "if (x > 5) { }",
        "g": 72
    },
    {
        "f": "algo_recur_75.png",
        "q": "What is a base case in recursion?",
        "s": "To stop the loop.",
        "a": "The base case is the condition that stops\nthe recursion and returns a value directly\nwithout making another recursive call.",
        "g": 75
    },
    {
        "f": "sql_quotes_75.png",
        "q": "Select all users where name is Dan.",
        "s": "SELECT * FROM u WHERE name=\"Dan\"",
        "a": "SELECT * FROM users WHERE name = 'Dan';",
        "g": 75
    },
    {
        "f": "java_loop_60.png",
        "q": "Write a loop that runs exactly 5 times.",
        "s": "while(i<5) print(i); // Infinite loop",
        "a": "for(int i=0; i<5; i++) { System.out.println(i); }",
        "g": 60
    },
    {
        "f": "py_indent_65.png",
        "q": "If x equals 5, print yes.",
        "s": "if x==5:\nprint('yes') // Indentation error",
        "a": "if x == 5:\n    print('yes')",
        "g": 65
    },
    {
        "f": "sql_join_65.png",
        "q": "Join the orders table with the users table.",
        "s": "SELECT * FROM orders JOIN users; // No ON clause",
        "a": "SELECT * FROM orders JOIN users\nON orders.user_id = users.id;",
        "g": 65
    },
    {
        "f": "py_import_60.png",
        "q": "Import the math module in Python.",
        "s": "include math",
        "a": "import math",
        "g": 60
    },
    {
        "f": "gen_api_60.png",
        "q": "What does API stand for and what is it?",
        "s": "Application Interface.",
        "a": "API stands for Application Programming Interface.\nIt allows different programs to communicate\nwith each other through defined rules.",
        "g": 60
    },
    {
        "f": "py_range_68.png",
        "q": "Write a loop that prints numbers 0 to 4.",
        "s": "for i in range(5)\n print i",
        "a": "for i in range(5):\n    print(i)",
        "g": 68
    },
    {
        "f": "java_ret_68.png",
        "q": "Write a method that returns the integer 5.",
        "s": "void get() { return 5; }",
        "a": "int get() { return 5; }",
        "g": 68
    },
    {
        "f": "sql_form_55.png",
        "q": "Select all rows from the users table.",
        "s": "SELECT * FORM users",
        "a": "SELECT * FROM users;",
        "g": 55
    },
    {
        "f": "algo_stack_50.png",
        "q": "Is a Stack a FIFO data structure?",
        "s": "Yes, Stack is FIFO.",
        "a": "No. A Stack is LIFO (Last In First Out).\nFIFO is used by Queues.",
        "g": 50
    },
    {
        "f": "py_var_70.png",
        "q": "Set a variable named valid to True.",
        "s": "valid-user = True",
        "a": "valid = True",
        "g": 70
    },
    {
        "f": "os_deadlock_80.png",
        "q": "What is a Deadlock in operating systems?",
        "s": "When two processes wait for each other.",
        "a": "Deadlock occurs when two or more processes\neach wait for a resource held by the other,\ncausing all of them to block permanently.",
        "g": 80
    },
    {
        "f": "net_tcp_95.png",
        "q": "What is the difference between TCP and UDP?",
        "s": "TCP reliable, UDP fast.",
        "a": "TCP is reliable, ordered, and connection-based.\nUDP is fast, connectionless, with no\nguarantee of delivery or order.",
        "g": 95
    },
    {
        "f": "cpp_cout_85.png",
        "q": "Print the word Hello in C++.",
        "s": "cout << 'Hello'",
        "a": "std::cout << \"Hello\" << std::endl;",
        "g": 85
    },
]


def create_image_from_text(text, filename, folder):
    W, H = 800, 600
    img = Image.new('RGB', (W, H), color='white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()

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


# ==========================================
# Main
# ==========================================
print("Generating dataset...")

if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

new_rows = []
for item in dataset:
    create_image_from_text(item['q'], item['f'], QUESTIONS_DIR)
    create_image_from_text(item['s'], item['f'], SOLUTIONS_DIR)
    create_image_from_text(item['a'], item['f'], ANSWERS_DIR)
    new_rows.append({'filename': item['f'], 'grade': item['g']})
    print(f"Created: {item['f']} (grade: {item['g']})")

pd.DataFrame(new_rows).to_csv(CSV_FILE, index=False)
print(f"\nDone. Created {len(new_rows)} exam sets.")