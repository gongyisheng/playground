# vscode settings
## python linting
- install pylance (vscode extension)
- install ruff (vscode extension)
use following settings.json
```
{
    "python.analysis.extraPaths": ["."],
    "python.languageServer": "Pylance",
    "editor.semanticHighlighting.enabled": true,
    "ruff.path": [".venv/bin/ruff"],
    "ruff.lint.enable": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    }
}
```