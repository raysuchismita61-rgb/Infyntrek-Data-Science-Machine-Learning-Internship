import pandas as pd


def assign_task(task_hours, required_category):

    df = pd.read_csv("data/cleaned_tasks.csv")

    users = df[
        ["Assigned_User", "Current_Workload"]
    ].drop_duplicates()

    users = users.sort_values(
        by="Current_Workload"
    )

    print("\nAvailable Users:")

    print(users)

    selected_user = users.iloc[0]["Assigned_User"]

    print(
        f"\nRecommended User: {selected_user}"
    )

    return selected_user


if __name__ == "__main__":

    assign_task(
        task_hours=5,
        required_category="Development"
    )