class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append({"task": task, "completed": False})

    def view_tasks(self):
        for index, task in enumerate(self.tasks):
            status = "Done" if task["completed"] else "Not Done"
            print(f"{index + 1}. {task['task']} - {status}")

    def complete_task(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks[index]["completed"] = True
        else:
            print("Invalid task number.")

def main():
    todo_list = TodoList()

    while True:
        print("\n1. Add task\n2. View tasks\n3. Complete task\n4. Quit")
        choice = input("Choose an option: ")

        if choice == "1":
            task = input("Enter the task: ")
            todo_list.add_task(task)
        elif choice == "2":
            todo_list.view_tasks()
        elif choice == "3":
            task_number = int(input("Enter the task number to complete: ")) - 1
            todo_list.complete_task(task_number)
        elif choice == "4":
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
