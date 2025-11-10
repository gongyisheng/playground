import os
from typing import Any
from fastmcp import FastMCP

ROOT_DIR = os.path.expanduser("~")  # Limit access to home directory or change this to your sandbox

def safe_join(root: str, subpath: str) -> str:
    """Prevent directory traversal."""
    abs_path = os.path.abspath(os.path.join(root, subpath))
    if not abs_path.startswith(root):
        raise PermissionError("Access denied")
    return abs_path


# Core implementation functions (can be tested directly)
def list_files_impl(path: str = ".") -> dict[str, Any]:
    """List files and directories in a given path.

    Args:
        path: Relative path from ROOT_DIR to list files from (default: ".")

    Returns:
        Dictionary with 'files' key containing list of file/directory info
    """
    abs_path = safe_join(ROOT_DIR, path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError("Path not found")

    files = []
    for entry in os.scandir(abs_path):
        files.append({
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file"
        })

    return {"files": files}


def search_files_impl(path: str = ".", keyword: str = "") -> dict[str, Any]:
    """Search for files by keyword in their filename.

    Args:
        path: Relative path from ROOT_DIR to search in (default: ".")
        keyword: Keyword to search for in filenames (case-insensitive, default: "")

    Returns:
        Dictionary with 'matches' key containing list of matching file paths
    """
    keyword_lower = keyword.lower()
    abs_path = safe_join(ROOT_DIR, path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError("Path not found")

    matches = []
    for dirpath, _, filenames in os.walk(abs_path):
        for f in filenames:
            if keyword_lower in f.lower():
                matches.append(os.path.join(dirpath, f))

    return {"matches": matches}


def read_file_impl(path: str) -> dict[str, Any]:
    """Read the content of a file.

    Args:
        path: Relative path from ROOT_DIR to the file to read

    Returns:
        Dictionary with 'content' key containing the file content
    """
    if not path:
        raise ValueError("Missing 'path' argument")

    abs_path = safe_join(ROOT_DIR, path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError("File not found")

    if os.path.isdir(abs_path):
        raise ValueError("Path is a directory")

    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {"content": content}


# Create FastMCP server and register tools
mcp = FastMCP("Filesystem MCP Server")


@mcp.tool()
def list_files(path: str = ".") -> dict[str, Any]:
    """List files and directories in a given path.

    Args:
        path: Relative path from ROOT_DIR to list files from (default: ".")

    Returns:
        Dictionary with 'files' key containing list of file/directory info
    """
    return list_files_impl(path)


@mcp.tool()
def search_files(path: str = ".", keyword: str = "") -> dict[str, Any]:
    """Search for files by keyword in their filename.

    Args:
        path: Relative path from ROOT_DIR to search in (default: ".")
        keyword: Keyword to search for in filenames (case-insensitive, default: "")

    Returns:
        Dictionary with 'matches' key containing list of matching file paths
    """
    return search_files_impl(path, keyword)


@mcp.tool()
def read_file(path: str) -> dict[str, Any]:
    """Read the content of a file.

    Args:
        path: Relative path from ROOT_DIR to the file to read

    Returns:
        Dictionary with 'content' key containing the file content
    """
    return read_file_impl(path)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
