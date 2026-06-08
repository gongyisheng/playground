# Code Style

- Minimal code: only necessary logic, no over-engineering
- Inline functions used by only one caller into that caller
- Comment only when necessary; keep comments and docstrings short — let the code carry the logic
- Prefer readable names over terse abbreviations: `chunk` not `ck`, `chunk_id` not `cid`

# Dev Rules

- Verify new runnable code executes without errors (e.g., `python3 xxx.py`, `g++ && ./a.out`, `npm run build`)

# Writing Style

- Concise, dense, and accurate — tight bullets over paragraphs, focus on the big picture, no padding
- When uncertain, say so and give your best read instead of hedging
- Only include examples when necessary
- All visualizations (graphs, charts, diagrams) must use valid Markdown syntax

# Git Commit Messages

Use these prefixes:
- `[feat]`: new feature
- `[fix]`: bug fix
- `[chore]`: routine tasks, maintenance, or refactor
- `[docs]`: documentation only
- `[style]`: code formatting (no logic changes)
- `[perf]`: performance improvement
- `[test]`: adding or fixing tests
- `[build]`: build system or external dependencies
- `[ci]`: CI/CD configuration or scripts
- `[deps]`: dependency upgrades/downgrades