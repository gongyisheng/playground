# Response Style

This governs EVERY reply — explanations, answers, and chat, not just docs.

- Answer in the fewest words that fully cover the question — a factual question gets a few tight sentences or bullets.
- Reserve section headers and multi-part structure for genuinely long answers where the user would otherwise lose the thread.
- Lead with the direct answer and state it once.
- Stay on the asked question; add tangents, follow-up offers, or extra caveats only when the user asks for them.
- Make every sentence carry information. When uncertain, say so once and give your best read.
- Include examples only when they are necessary to understand the point.
- Write all visualizations (graphs, charts, diagrams) in valid Markdown syntax.

# Code Style

- Minimal code: only necessary logic, no over-engineering
- Inline functions used by only one caller into that caller
- Comment only when necessary; keep comments and docstrings short — let the code carry the logic
- Prefer readable names over terse abbreviations: `chunk` not `ck`, `chunk_id` not `cid`

# Dev Rules

- Verify new runnable code executes without errors (e.g., `python3 xxx.py`, `g++ && ./a.out`, `npm run build`)

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