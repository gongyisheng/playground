from typing import Optional

# File viewer

def open(path: str, line_number: Optional[int] = None) -> None:
    """
    Opens the file at the given path in the editor. 
    If line_number is provided, the window will move to include that line.
    """
    pass


def goto(line_number: int) -> None:
    """
    Moves the window to show line_number.
    """
    pass


def scroll_down() -> None:
    """
    Moves the window up 100 lines.
    """
    pass


def scroll_up() -> None:
    """
    Moves the window down 100 lines.
    """
    pass

# Search Tools

def search_file(search_term: str, file: Optional[str] = None) -> None:
    """
    Searches for search_term in file. 
    If file is not provided, searches in the current open file.
    """
    pass


def search_dir(search_term: str, dir: Optional[str] = None) -> None:
    """
    Searches for search_term in all files in dir. 
    If dir is not provided, searches in the current directory.
    """
    pass


def find_file(file_name: str, dir: Optional[str] = None) -> None:
    """
    Finds all files with the given name in dir. 
    If dir is not provided, searches in the current directory.
    """
    pass

# File Editing

def edit(line_range: str, replacement_text: str) -> None:
    """
    Replaces lines n through m (inclusive) with the given text in the open file.
    All of the replacement_text will be entered, so make sure your indentation
    is formatted properly. Python files will be checked for syntax errors after
    the edit. If an error is found, the edit will not be executed.
    
    Args:
        line_range: Format "n:m" where n and m are line numbers
        replacement_text: Text to replace the specified lines
    """
    pass


def create(filename: str) -> None:
    """
    Creates and opens a new file with the given name.
    """
    pass

# Task

def submit() -> None:
    """
    Generates and submits the patch from all previous edits and closes the shell.
    """
    pass