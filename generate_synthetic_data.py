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
    # ==========================================
    # NEW SAMPLES - better grade distribution
    # ==========================================
    {
        "f": "py_half_50.png",
        "q": "Write a function that returns the maximum of two numbers.",
        "s": "def max(a, b):",
        "a": "def max(a, b):\n    if a > b:\n        return a\n    return b",
        "g": 50
    },
    {
        "f": "java_half_50.png",
        "q": "Write a method that checks if a number is even.",
        "s": "boolean isEven(int n) { }",
        "a": "boolean isEven(int n) {\n    return n % 2 == 0;\n}",
        "g": 50
    },
    {
        "f": "sql_half_50.png",
        "q": "Write a query that counts the number of users in the users table.",
        "s": "SELECT FROM users;",
        "a": "SELECT COUNT(*) FROM users;",
        "g": 50
    },
    {
        "f": "algo_half_50.png",
        "q": "What are the two main properties of a Binary Search Tree?",
        "s": "Left node is smaller than root.",
        "a": "Left subtree contains only nodes smaller than root.\nRight subtree contains only nodes greater than root.",
        "g": 50
    },
    {
        "f": "py_zero_0.png",
        "q": "Write a Python function that returns the square of a number.",
        "s": "def square: x * x",
        "a": "def square(x):\n    return x * x",
        "g": 0
    },
    {
        "f": "java_zero_0.png",
        "q": "Declare a string variable named name with value John in Java.",
        "s": "int name = John;",
        "a": "String name = \"John\";",
        "g": 0
    },
    {
        "f": "sql_zero_0.png",
        "q": "Select only the name column from the users table.",
        "s": "UPDATE users SET name;",
        "a": "SELECT name FROM users;",
        "g": 0
    },
    {
        "f": "algo_zero_0.png",
        "q": "What is the time complexity of linear search?",
        "s": "O(1) because it checks one element.",
        "a": "O(n) because in the worst case it checks every element.",
        "g": 0
    },
    {
        "f": "py_perfect2_100.png",
        "q": "Write a Python function that reverses a string.",
        "s": "def reverse(s):\n    return s[::-1]",
        "a": "def reverse(s):\n    return s[::-1]",
        "g": 100
    },
    {
        "f": "java_perfect2_100.png",
        "q": "Write a Java method that returns the absolute value of an integer.",
        "s": "int abs(int n) { return n < 0 ? -n : n; }",
        "a": "int abs(int n) { return n < 0 ? -n : n; }",
        "g": 100
    },
    {
        "f": "sql_perfect2_100.png",
        "q": "Select all columns from the products table where price is greater than 100.",
        "s": "SELECT * FROM products WHERE price > 100;",
        "a": "SELECT * FROM products WHERE price > 100;",
        "g": 100
    },
    {
        "f": "algo_perfect2_100.png",
        "q": "What is the time complexity of binary search?",
        "s": "O(log n) because it halves the search space each step.",
        "a": "O(log n) because the array is halved at each step.",
        "g": 100
    },
    {
        "f": "py_mostly_80.png",
        "q": "Write a Python loop that prints even numbers from 0 to 10.",
        "s": "for i in range(0, 10, 2):\n    print(i)",
        "a": "for i in range(0, 11, 2):\n    print(i)",
        "g": 80
    },
    {
        "f": "java_mostly_80.png",
        "q": "Write a Java loop that prints numbers 1 to 10.",
        "s": "for(int i=1; i<=10; i++) System.out.print(i);",
        "a": "for(int i=1; i<=10; i++) System.out.println(i);",
        "g": 80
    },
    {
        "f": "sql_mostly_80.png",
        "q": "Select all users whose age is greater than 18, ordered by name.",
        "s": "SELECT * FROM users WHERE age > 18;",
        "a": "SELECT * FROM users WHERE age > 18 ORDER BY name;",
        "g": 80
    },
    {
        "f": "algo_mostly_80.png",
        "q": "Explain what a Queue data structure is.",
        "s": "A Queue is FIFO. Elements are added at the back.",
        "a": "A Queue is FIFO (First In First Out).\nElements are added at the back and removed from the front.",
        "g": 80
    },
    {
        "f": "py_low_25.png",
        "q": "Write a Python class named Animal with a method speak that prints Animal speaks.",
        "s": "class Animal",
        "a": "class Animal:\n    def speak(self):\n        print('Animal speaks')",
        "g": 25
    },
    {
        "f": "java_low_25.png",
        "q": "Write a Java interface named Shape with a method getArea that returns a double.",
        "s": "interface Shape { }",
        "a": "interface Shape {\n    double getArea();\n}",
        "g": 25
    },
    {
        "f": "sql_low_25.png",
        "q": "Write a query that finds the average salary from the employees table.",
        "s": "SELECT salary FROM employees;",
        "a": "SELECT AVG(salary) FROM employees;",
        "g": 25
    },
    {
        "f": "algo_low_25.png",
        "q": "What is the difference between BFS and DFS?",
        "s": "BFS uses a queue.",
        "a": "BFS uses a Queue and explores level by level.\nDFS uses a Stack and explores as deep as possible first.",
        "g": 25
    },
    {
        "f": "py_near_95.png",
        "q": "Write a Python function that checks if a string is a palindrome.",
        "s": "def is_palindrome(s):\n    return s == s[::-1]",
        "a": "def is_palindrome(s):\n    return s == s[::-1]",
        "g": 95
    },
    {
        "f": "java_near_95.png",
        "q": "Write a Java method that returns true if a number is positive.",
        "s": "boolean isPositive(int n) { return n > 0; }",
        "a": "boolean isPositive(int n) { return n > 0; }",
        "g": 95
    },
    {
        "f": "os_half_50.png",
        "q": "What are the four conditions required for deadlock?",
        "s": "Mutual exclusion and hold and wait.",
        "a": "Mutual exclusion, hold and wait,\nno preemption, and circular wait.",
        "g": 50
    },
    {
        "f": "net_half_50.png",
        "q": "What are the three steps of the TCP three-way handshake?",
        "s": "SYN and SYN-ACK.",
        "a": "SYN sent by client,\nSYN-ACK sent by server,\nACK sent by client.",
        "g": 50
    },
    {
        "f": "os_zero_0.png",
        "q": "What does CPU scheduling decide?",
        "s": "It manages the hard drive.",
        "a": "CPU scheduling decides which process runs next\nand for how long, to maximize CPU utilization.",
        "g": 0
    },
    {
        "f": "net_zero_0.png",
        "q": "What is the purpose of DNS?",
        "s": "DNS encrypts internet traffic.",
        "a": "DNS translates domain names into IP addresses\nso computers can find each other on a network.",
        "g": 0
    },
    {
        "f": "os_perfect_100.png",
        "q": "What is virtual memory?",
        "s": "Virtual memory allows programs to use more memory than physically available by using disk space as an extension of RAM.",
        "a": "Virtual memory uses disk space to extend RAM,\nallowing programs to run even when physical memory is full.",
        "g": 100
    },
    {
        "f": "net_perfect_100.png",
        "q": "What does HTTP stand for and what is it used for?",
        "s": "HTTP stands for HyperText Transfer Protocol. It is used to transfer web pages over the internet.",
        "a": "HTTP stands for HyperText Transfer Protocol.\nIt is used to transfer data between web browsers and servers.",
        "g": 100
    },
    {
        "f": "algo_near100_95.png",
        "q": "What is the space complexity of merge sort?",
        "s": "O(n) because it needs extra space for the temporary arrays.",
        "a": "O(n) auxiliary space is needed for the temporary arrays during merging.",
        "g": 95
    },
    {
        "f": "py_near100_95.png",
        "q": "What is a decorator in Python?",
        "s": "A decorator is a function that wraps another function to add behavior without changing it.",
        "a": "A decorator is a function that takes another function\nand extends its behavior without modifying it directly.",
        "g": 95
    },
    {
        "f": "py_mid_63.png",
        "q": "Write a Python function that returns the factorial of n recursively.",
        "s": "def factorial(n):\n    return n * factorial(n-1)",
        "a": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
        "g": 63
    },
    {
        "f": "java_mid_71.png",
        "q": "Write a Java method that takes an array and returns its sum.",
        "s": "int sum(int[] arr) {\n    int total = 0;\n    for(int x : arr) total += x;\n    return total;\n}",
        "a": "int sum(int[] arr) {\n    int total = 0;\n    for(int x : arr) total += x;\n    return total;\n}",
        "g": 71
    },
    {
        "f": "sql_mid_67.png",
        "q": "Select the names and emails of users who registered after 2020.",
        "s": "SELECT name FROM users WHERE year > 2020;",
        "a": "SELECT name, email FROM users WHERE YEAR(registered) > 2020;",
        "g": 67
    },
    {
        "f": "algo_mid_74.png",
        "q": "What is the worst case time complexity of bubble sort and why?",
        "s": "O(n^2) because of nested loops.",
        "a": "O(n^2) because for each of the n elements,\nit compares with all remaining elements using nested loops.",
        "g": 74
    },
    {
        "f": "py_mid_46.png",
        "q": "Write a Python class Dog with a constructor that sets the name attribute.",
        "s": "class Dog:\n    def __init__(self):\n        pass",
        "a": "class Dog:\n    def __init__(self, name):\n        self.name = name",
        "g": 46
    },
    {
        "f": "java_mid_43.png",
        "q": "Write a Java while loop that prints numbers 1 to 5.",
        "s": "while(true) { System.out.println(1); }",
        "a": "int i = 1;\nwhile(i <= 5) {\n    System.out.println(i);\n    i++;\n}",
        "g": 43
    },
    {
        "f": "sql_mid_77.png",
        "q": "Insert a new user with name Alice and age 25 into the users table.",
        "s": "INSERT INTO users (name, age) VALUES ('Alice', 25)",
        "a": "INSERT INTO users (name, age) VALUES ('Alice', 25);",
        "g": 77
    },
    {
        "f": "os_mid_62.png",
        "q": "What is the difference between a process and a thread in terms of memory?",
        "s": "Processes have separate memory. Threads share memory.",
        "a": "Each process has its own separate memory space.\nThreads within the same process share the same memory space\nand can communicate directly.",
        "g": 62
    },
    {
        "f": "net_mid_69.png",
        "q": "What is the difference between HTTP and HTTPS?",
        "s": "HTTPS is secure and encrypted, HTTP is not.",
        "a": "HTTP sends data in plain text with no encryption.\nHTTPS uses SSL/TLS to encrypt data,\nmaking it secure against interception.",
        "g": 69
    },
    {
        "f": "algo_mid_42.png",
        "q": "Explain what a linked list is.",
        "s": "A list of elements.",
        "a": "A linked list is a data structure where each node\ncontains data and a pointer to the next node.\nUnlike arrays, elements are not stored contiguously in memory.",
        "g": 42
    },
    {
        "f": "py_mid_73.png",
        "q": "Write a Python function that takes a list and returns only the even numbers.",
        "s": "def get_evens(lst):\n    return [x for x in lst if x % 2 == 0]",
        "a": "def get_evens(lst):\n    return [x for x in lst if x % 2 == 0]",
        "g": 73
    },
    {
        "f": "java_mid_66.png",
        "q": "What is method overloading in Java?",
        "s": "When two methods have the same name.",
        "a": "Method overloading is when two or more methods\nhave the same name but different parameter lists.\nJava decides which to call based on the arguments passed.",
        "g": 66
    },
    {
        "f": "sql_mid_44.png",
        "q": "Write a query to delete all users older than 60 from the users table.",
        "s": "DELETE FROM users;",
        "a": "DELETE FROM users WHERE age > 60;",
        "g": 44
    },
    {
        "f": "os_mid_76.png",
        "q": "What is a semaphore used for in operating systems?",
        "s": "A semaphore controls access to shared resources\nto prevent race conditions.",
        "a": "A semaphore is a synchronization tool used to control\naccess to shared resources by multiple processes,\npreventing race conditions.",
        "g": 76
    },
    {
        "f": "net_mid_61.png",
        "q": "What is a MAC address?",
        "s": "A unique address assigned to a network device.",
        "a": "A MAC address is a unique hardware identifier\nassigned to a network interface card.\nIt operates at the data link layer and is used for local network communication.",
        "g": 61
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