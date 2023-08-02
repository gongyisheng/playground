import curses

def main(stdscr):
    # Setup
    curses.curs_set(0)
    stdscr.clear()

    # Data structure to store the tasks
    tasks = []

    # Main loop
    while True:
        stdscr.addstr(0, 0, "To-Do List App", curses.A_BOLD)
        stdscr.addstr(2, 0, "Press 'a' to add a new task.")
        stdscr.addstr(3, 0, "Press 'q' to quit.")

        # Display tasks
        for i, task in enumerate(tasks):
            stdscr.addstr(5 + i, 0, f"{i+1}. {task}")

        # Wait for user input
        key = stdscr.getch()

        # Handle user input
        if key == ord('a'):
            stdscr.addstr(7 + len(tasks), 0, "Enter task: ")
            stdscr.refresh()
            curses.echo()  # Enable text input
            new_task = stdscr.getstr(7 + len(tasks), len("Enter task: "))
            curses.noecho()  # Disable text input

            if new_task:
                tasks.append(new_task.decode("utf-8"))

        elif key == ord('q'):
            break

        stdscr.clear()

if __name__ == "__main__":
    curses.wrapper(main)
