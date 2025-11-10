import os
import tempfile
import shutil
import pytest
from server import list_files_impl as list_files, search_files_impl as search_files, read_file_impl as read_file, ROOT_DIR


@pytest.fixture
def test_files():
    """Create temporary test files and directories."""
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(dir=ROOT_DIR)
    test_subdir = os.path.join(test_dir, "subdir")
    os.makedirs(test_subdir)

    # Create test files
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Hello, World!")

    test_file2 = os.path.join(test_subdir, "example.txt")
    with open(test_file2, "w") as f:
        f.write("Example content")

    # Get relative paths from ROOT_DIR
    rel_test_dir = os.path.relpath(test_dir, ROOT_DIR)
    rel_test_file = os.path.relpath(test_file, ROOT_DIR)

    yield {
        "test_dir": test_dir,
        "test_subdir": test_subdir,
        "test_file": test_file,
        "test_file2": test_file2,
        "rel_test_dir": rel_test_dir,
        "rel_test_file": rel_test_file,
    }

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


class TestMCPTools:
    """Test cases for the MCP filesystem server tools."""

    def test_list_files(self, test_files):
        """Test listing files in a directory."""
        result = list_files(path=test_files["rel_test_dir"])

        assert "files" in result

        # Check that our test files are listed
        file_names = [f["name"] for f in result["files"]]
        assert "test.txt" in file_names
        assert "subdir" in file_names

        # Check file types
        for f in result["files"]:
            if f["name"] == "test.txt":
                assert f["type"] == "file"
            elif f["name"] == "subdir":
                assert f["type"] == "directory"

    def test_list_files_default_path(self):
        """Test listing files with default path."""
        result = list_files()
        assert "files" in result
        assert isinstance(result["files"], list)

    def test_list_files_nonexistent_path(self):
        """Test listing files in a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            list_files(path="nonexistent/path")

    def test_search_files(self, test_files):
        """Test searching for files by keyword."""
        result = search_files(
            path=test_files["rel_test_dir"],
            keyword="test"
        )

        assert "matches" in result

        # Should find test.txt
        matches = result["matches"]
        assert any("test.txt" in match for match in matches)

    def test_search_files_with_different_case(self, test_files):
        """Test case-insensitive search."""
        result = search_files(
            path=test_files["rel_test_dir"],
            keyword="EXAMPLE"
        )

        assert "matches" in result

        # Should find example.txt despite case difference
        matches = result["matches"]
        assert any("example.txt" in match for match in matches)

    def test_search_files_no_matches(self, test_files):
        """Test search with no matching files."""
        result = search_files(
            path=test_files["rel_test_dir"],
            keyword="nonexistent"
        )

        assert "matches" in result
        assert len(result["matches"]) == 0

    def test_search_files_nonexistent_path(self):
        """Test searching in a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            search_files(path="nonexistent/path", keyword="test")

    def test_read_file(self, test_files):
        """Test reading file content."""
        result = read_file(path=test_files["rel_test_file"])

        assert "content" in result
        assert result["content"] == "Hello, World!"

    def test_read_file_from_subdir(self, test_files):
        """Test reading file from subdirectory."""
        rel_test_file2 = os.path.relpath(test_files["test_file2"], ROOT_DIR)
        result = read_file(path=rel_test_file2)

        assert "content" in result
        assert result["content"] == "Example content"

    def test_read_file_missing_path(self):
        """Test read_file with missing path argument."""
        with pytest.raises(ValueError, match="Missing 'path' argument"):
            read_file(path="")

    def test_read_file_nonexistent(self):
        """Test reading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_file(path="nonexistent.txt")

    def test_read_file_directory(self, test_files):
        """Test reading a directory (should fail)."""
        with pytest.raises(ValueError, match="Path is a directory"):
            read_file(path=test_files["rel_test_dir"])

    def test_directory_traversal_prevention(self):
        """Test that directory traversal is prevented."""
        with pytest.raises(PermissionError):
            list_files(path="../../etc")

        with pytest.raises(PermissionError):
            search_files(path="../../etc", keyword="passwd")

        with pytest.raises(PermissionError):
            read_file(path="../../etc/passwd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
