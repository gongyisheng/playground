import subprocess

def run_sysbench():
    # Define the sysbench command
    command = ["sysbench", "--test=cpu", "--cpu-max-prime=20000", "run"]

    # Execute the sysbench command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output
    if result.returncode == 0:
        print("Sysbench CPU Test Output:\n" + result.stdout)
    else:
        print("An error occurred while executing sysbench:\n" + result.stderr)

if __name__ == "__main__":
    run_sysbench()