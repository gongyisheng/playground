from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', help='dir', type=str, required=True)
    parser.add_argument('-f', '--file', dest='file', help='file', type=str, required=True)
    args = parser.parse_args()
    return args

# check the difference between two files 
# using built-in diff command of terminal
# directory structure:
# |<dir_name>-prod
# --|<file_name>
# |<dir_name>-stag
# --|<file_name>
def print_command(dir_name, file_name):
    command = f"diff {dir_name}-prod/{file_name} {dir_name}-stag/{file_name}"
    print(command)

if __name__ == "__main__":
    args = parse_args()
    dir_name = args.dir
    file_name = args.file
    print_command(dir_name, file_name)