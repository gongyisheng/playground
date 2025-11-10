import os
import json
import tempfile
import shutil
import pytest
from tornado.httpclient import HTTPRequest
from tornado.testing import AsyncHTTPTestCase
from server import make_app, ROOT_DIR


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


class TestInvokeHandler(AsyncHTTPTestCase):
    """Test cases for the MCP filesystem server."""

    def get_app(self):
        """Create and return the Tornado application."""
        return make_app()

    def test_list_files(self):
        """Test listing files in a directory."""
        # Setup
        test_dir = tempfile.mkdtemp(dir=ROOT_DIR)
        test_subdir = os.path.join(test_dir, "subdir")
        os.makedirs(test_subdir)

        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, World!")

        rel_test_dir = os.path.relpath(test_dir, ROOT_DIR)

        try:
            payload = {
                "resource": "list_files",
                "args": {"path": rel_test_dir}
            }

            response = self.fetch(
                "/invoke",
                method="POST",
                body=json.dumps(payload)
            )

            assert response.code == 200
            data = json.loads(response.body)
            assert "files" in data

            # Check that our test files are listed
            file_names = [f["name"] for f in data["files"]]
            assert "test.txt" in file_names
            assert "subdir" in file_names

            # Check file types
            for f in data["files"]:
                if f["name"] == "test.txt":
                    assert f["type"] == "file"
                elif f["name"] == "subdir":
                    assert f["type"] == "directory"
        finally:
            # Cleanup
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_search_files(self):
        """Test searching for files by keyword."""
        # Setup
        test_dir = tempfile.mkdtemp(dir=ROOT_DIR)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello")

        rel_test_dir = os.path.relpath(test_dir, ROOT_DIR)

        try:
            payload = {
                "resource": "search_files",
                "args": {
                    "path": rel_test_dir,
                    "keyword": "test"
                }
            }

            response = self.fetch(
                "/invoke",
                method="POST",
                body=json.dumps(payload)
            )

            assert response.code == 200
            data = json.loads(response.body)
            assert "matches" in data

            # Should find test.txt
            matches = data["matches"]
            assert any("test.txt" in match for match in matches)
        finally:
            # Cleanup
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_search_files_with_different_case(self):
        """Test case-insensitive search."""
        # Setup
        test_dir = tempfile.mkdtemp(dir=ROOT_DIR)
        test_subdir = os.path.join(test_dir, "subdir")
        os.makedirs(test_subdir)

        test_file2 = os.path.join(test_subdir, "example.txt")
        with open(test_file2, "w") as f:
            f.write("Example content")

        rel_test_dir = os.path.relpath(test_dir, ROOT_DIR)

        try:
            payload = {
                "resource": "search_files",
                "args": {
                    "path": rel_test_dir,
                    "keyword": "EXAMPLE"
                }
            }

            response = self.fetch(
                "/invoke",
                method="POST",
                body=json.dumps(payload)
            )

            assert response.code == 200
            data = json.loads(response.body)
            assert "matches" in data

            # Should find example.txt despite case difference
            matches = data["matches"]
            assert any("example.txt" in match for match in matches)
        finally:
            # Cleanup
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_read_file(self):
        """Test reading file content."""
        # Setup
        test_dir = tempfile.mkdtemp(dir=ROOT_DIR)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, World!")

        rel_test_file = os.path.relpath(test_file, ROOT_DIR)

        try:
            payload = {
                "resource": "read_file",
                "args": {"path": rel_test_file}
            }

            response = self.fetch(
                "/invoke",
                method="POST",
                body=json.dumps(payload)
            )

            assert response.code == 200
            data = json.loads(response.body)
            assert "content" in data
            assert data["content"] == "Hello, World!"
        finally:
            # Cleanup
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_read_file_missing_path(self):
        """Test read_file with missing path argument."""
        payload = {
            "resource": "read_file",
            "args": {}
        }

        response = self.fetch(
            "/invoke",
            method="POST",
            body=json.dumps(payload)
        )

        assert response.code == 500
        data = json.loads(response.body)
        assert "error" in data

    def test_unknown_resource(self):
        """Test handling of unknown resource."""
        payload = {
            "resource": "unknown_operation",
            "args": {}
        }

        response = self.fetch(
            "/invoke",
            method="POST",
            body=json.dumps(payload)
        )

        assert response.code == 400
        data = json.loads(response.body)
        assert "error" in data
        assert "Unknown resource" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
