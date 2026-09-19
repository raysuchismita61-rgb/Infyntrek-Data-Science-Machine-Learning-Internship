import pandas as pd
import random
import os

random.seed(42)

task_templates = {
    "Bug": [
        "Fix login authentication bug",
        "Resolve payment processing error",
        "Fix database connection issue",
        "Debug API response failure",
        "Resolve application crash",
        "Fix user registration problem",
        "Correct password reset issue"
    ],

    "Development": [
        "Develop user authentication module",
        "Create REST API for products",
        "Implement dashboard functionality",
        "Develop payment gateway",
        "Build task management module",
        "Create user profile page",
        "Implement notification system"
    ],

    "Testing": [
        "Test login functionality",
        "Perform API testing",
        "Test payment module",
        "Check application security",
        "Perform regression testing",
        "Test user registration",
        "Validate dashboard functionality"
    ],

    "Documentation": [
        "Update project documentation",
        "Write API documentation",
        "Prepare user manual",
        "Document database structure",
        "Update installation guide",
        "Write technical documentation"
    ],

    "Design": [
        "Design login page",
        "Create dashboard UI",
        "Design user profile interface",
        "Improve website layout",
        "Create application prototype",
        "Design mobile interface"
    ]
}

priorities = ["Low", "Medium", "High", "Critical"]

users = [
    "Developer_1",
    "Developer_2",
    "Tester_1",
    "Designer_1",
    "Manager_1"
]

rows = []

for i in range(500):

    category = random.choice(list(task_templates.keys()))

    description = random.choice(task_templates[category])

    priority = random.choices(
        priorities,
        weights=[20, 35, 30, 15]
    )[0]

    days_remaining = random.randint(1, 30)

    estimated_hours = random.randint(1, 20)

    workload = random.randint(10, 100)

    assigned_user = random.choice(users)

    rows.append({
        "Task_ID": i + 1,
        "Task_Description": description,
        "Category": category,
        "Priority": priority,
        "Days_Remaining": days_remaining,
        "Estimated_Hours": estimated_hours,
        "Current_Workload": workload,
        "Assigned_User": assigned_user
    })

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)

df.to_csv("data/tasks.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())