import os
import json
import tornado.ioloop
import tornado.web

ROOT_DIR = os.path.expanduser("~")  # Limit access to home directory or change this to your sandbox

def safe_join(root, subpath):
    """Prevent directory traversal."""
    abs_path = os.path.abspath(os.path.join(root, subpath))
    if not abs_path.startswith(root):
        raise PermissionError("Access denied")
    return abs_path


class InvokeHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            # Parse request JSON
            data = json.loads(self.request.body.decode("utf-8"))
            resource = data.get("resource")
            args = data.get("args", {})

            if resource == "list_files":
                result = await self.list_files(args)
            elif resource == "search_files":
                result = await self.search_files(args)
            elif resource == "read_file":
                result = await self.read_file(args)
            else:
                self.set_status(400)
                self.write({"error": f"Unknown resource: {resource}"})
                return

            self.write(result)

        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

    async def list_files(self, args):
        path_arg = args.get("path", ".")
        abs_path = safe_join(ROOT_DIR, path_arg)

        if not os.path.exists(abs_path):
            raise FileNotFoundError("Path not found")

        files = []
        for entry in os.scandir(abs_path):
            files.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file"
            })

        return {"files": files}

    async def search_files(self, args):
        path_arg = args.get("path", ".")
        keyword = args.get("keyword", "").lower()
        abs_path = safe_join(ROOT_DIR, path_arg)

        if not os.path.exists(abs_path):
            raise FileNotFoundError("Path not found")

        matches = []
        for dirpath, _, filenames in os.walk(abs_path):
            for f in filenames:
                if keyword in f.lower():
                    matches.append(os.path.join(dirpath, f))

        return {"matches": matches}

    async def read_file(self, args):
        path_arg = args.get("path")
        if not path_arg:
            raise ValueError("Missing 'path' argument")

        abs_path = safe_join(ROOT_DIR, path_arg)

        if not os.path.exists(abs_path):
            raise FileNotFoundError("File not found")

        if os.path.isdir(abs_path):
            raise ValueError("Path is a directory")

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {"content": content}


def make_app():
    return tornado.web.Application([
        (r"/invoke", InvokeHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    port = 8001
    print(f"Filesystem MCP server running on http://localhost:{port}")
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()
