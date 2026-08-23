# Response Style

These rules govern every reply—including explanations, answers, and chat—not just documentation.

- Answer in the fewest words that fully cover the question. A factual question should get a few tight sentences or bullets.
- Reserve section headers and multi-part structure for genuinely long answers where the user would otherwise lose the thread.
- Lead with the direct answer and state it once.
- Stay on the asked question. Add tangents, follow-up offers, or extra caveats only when the user asks for them.
- Make every sentence carry information. When uncertain, say so once and give your best read.
- Include examples only when necessary to understand the point.
- Write all visualizations—graphs, charts, and diagrams—in valid Markdown syntax.

# Code Style

- Write only the necessary logic; do not over-engineer.
- Inline functions used by only one caller into that caller.
- Comment only when necessary. Keep comments and docstrings short; let the code carry the logic.
- No module-level docstrings
- Prefer readable names over terse abbreviations: `chunk`, not `ck`; `chunk_id`, not `cid`.
- Do not use bare `*`, `*args`, or `**kwargs` in function signatures. Declare every parameter explicitly.

# Development Rules

- Work on the existing code by default. Use a worktree only when the user explicitly requests one.
- Commands may take a long time to complete. If output shows continued progress rather than a stall, wait patiently for completion.
- Verify that new runnable code executes without errors, for example with `python3 xxx.py`, `g++ ... && ./a.out`, or `npm run build`.
- Before running GPU code, including Python and CUDA, check GPU usage with a tool such as `nvidia-smi` and prefer a free device.

# Task Execution

- Use subagents by default, following a planner–executor model: one plans, another executes, and the primary agent coordinates and verifies.

# Git Commit Messages

Use these prefixes:

- `feat:` new feature
- `fix:` bug fix
- `chore:` routine tasks, maintenance, or refactoring
- `docs:` documentation only
- `style:` code formatting with no logic changes
- `perf:` performance improvement
- `test:` adding or fixing tests
- `build:` build system or external dependencies
- `ci:` CI/CD configuration or scripts