# Bub — High-Level Design

Bub is a **collaborative agent framework** for shared delivery workflows. Built on [Republic](https://github.com/bubbuild/republic) (tape-based deterministic LLM execution), it treats context as explicit assembly from verifiable interaction history — designed for teams that need inspectable, auditable, and handoff-friendly AI workflows.

**Version:** 0.2.3 | **Python:** >=3.12 | **Key deps:** Pydantic, Republic, Typer, Loguru, APScheduler, any-llm-sdk

---

## Directory Structure

```
src/bub/
├── app/            # Application runtime and bootstrap
├── channels/       # External channel adapters (Telegram, Discord)
├── cli/            # CLI interface (interactive shell, one-off runner)
├── config/         # Pydantic settings management
├── core/           # Core agent loop and routing
├── integrations/   # External system integration (Republic LLM client)
├── skills/         # Extensible agent capabilities (YAML frontmatter + markdown)
├── tape/           # Append-only interaction history (JSONL-based)
└── tools/          # Tool registry and progressive view
```

---

## Module-by-Module Breakdown

### 1. Core Agent Loop (`core/`)

The deterministic, single-session interaction loop. The heart of Bub.

**Components:**

- **`AgentLoop`** — Orchestrates one cycle: user input → routing → model response → routing. Returns `LoopResult` (visible text, exit flag, step count, errors).

- **`InputRouter`** — Dual-mode routing: applies the *same* command parsing rules to both user input and assistant output. User routing detects `,`-prefixed internal commands or shell commands; assistant routing parses fenced code blocks for executable commands. Failed commands are wrapped in `<command>` blocks and forwarded to the model as context.

- **`ModelRunner`** — Bounded loop (max 20 steps) for model turns. Key features:
  - **Skill hints**: `$skill-name` in text triggers expanded skill details
  - **Progressive tool expansion**: sends minimal summaries; full schemas on demand
  - **Follow-up handling**: if model calls tools, sends "Continue the task." for next turn
  - System prompt assembly: base prompt + workspace prompt (`AGENTS.md`) + runtime contract + tool/skill descriptions + token usage stats

- **`CommandDetector`** — Detects shell commands vs natural language using `shutil.which()`, path detection, and env variable prefix detection.

**Data flow for a single turn:**
```
User Input
  → InputRouter.route_user()
    ├─ Comma command? → Execute directly (success: return, fail: wrap for model)
    └─ Natural language → ModelRunner.run()
        → Loop: tape.run_tools_async() → route_assistant() → extract commands
            → Execute commands → collect results → continue or exit
  → LoopResult → Output to user
```

---

### 2. Tape System (`tape/`)

Append-only, verifiable interaction history with deterministic replay.

- **`TapeService`** — High-level API: handoff/anchor (phase boundaries with state capture), append events, fork/merge (temporary tape branches per user input with automatic rollback), search (fuzzy matching on entries), info (metadata query).

- **`FileTapeStore`** — JSONL-based persistent storage in `~/.bub/tapes/`. Thread-safe with file locking, supports copying/forking/merging, incremental reads (caches offset). Workspace-specific tape paths via MD5 hashing.

- **`AnchorSummary`** — Minimal dataclass for anchor metadata (name, state dict). Anchors mark checkpoints for phase transitions and handoffs.

**Design:** One session = one tape. The tape is the single source of truth for all I/O — enabling audit, replay, and safe rollback via forked tapes per user input.

---

### 3. Application Runtime (`app/`)

Global orchestration layer.

- **`AppRuntime`** — Singleton managing: settings, multiple `SessionRuntime` instances (one per session_id), shared `ToolRegistry`, APScheduler for scheduled tasks, and `ChannelManager` for multi-protocol support. Key methods: `get_session()` (lazy session creation), `handle_input()` (main entry point), `graceful_shutdown()`, `discover_skills()`.

- **`SessionRuntime`** — Per-session state: own `AgentLoop`, `TapeService`, `ModelRunner`, and `ToolView`. Sessions are isolated via forked tapes to prevent state leakage between users.

- **`build_runtime()`** — Factory function to create `AppRuntime` from `.env` and environment variables.

- **`BubJobStore`** — APScheduler persistent job storage for scheduled tasks.

**Multi-session architecture:**
```
AppRuntime (global)
  ├─ Settings, ToolRegistry, Scheduler
  ├─ Sessions (per session_id)
  │   └─ SessionRuntime → AgentLoop, TapeService, ModelRunner, ToolView
  └─ ChannelManager (optional)
```

---

### 4. Tools System (`tools/`)

Unified registry for all executable actions.

- **`ToolRegistry`** — Decorator-based registration (`@register(name, short_description, model)`). Filters by `allowed_tools`, wraps calls with logging/timing, generates Republic-compatible `Tool` schemas.

- **Built-in tools** (`builtin.py`):
  - `bash` — Shell execution with timeout and env loading
  - `fs.read / fs.write / fs.edit` — File operations
  - `web.fetch / web.search` — HTTP GET and Tavily-backed search
  - `tape.info / tape.search / tape.reset` — Tape operations
  - `skills.list / skills.describe` — Skill catalog
  - `schedule.add / schedule.remove / schedule.list` — APScheduler integration
  - `handoff` — Create anchor with summary + next steps
  - `help / tools / quit` — Internal commands

- **`ProgressiveToolView`** — Smart tool detail expansion. Tracks which tools are actually used (`note_selected()`) and hinted (`note_hint()`). Renders frequent tools in full detail, others in compact form — saving tokens in the system prompt.

---

### 5. Skills System (`skills/`)

Extensible agent capabilities loaded from project/global/builtin directories.

- **Discovery**: scans `.agent/skills/` (project), `~/.agent/skills/` (global), and builtin directories. Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:
  ```yaml
  ---
  name: friendly-python
  description: Write clean Python code
  metadata:
    tags: [python, code]
  ---
  # Skill body (detailed instructions)
  ```

- **Source hierarchy**: project > global > builtin (first match wins, case-insensitive).

- **Lazy activation**: Skills appear as compact summaries in the system prompt. The `$skill-name` syntax in model output triggers expansion of full skill details — avoids bloating the prompt with unused skill docs.

- **Bundled skills**: GitHub integration (`gh/`), Telegram/Discord channel skills, and specialized skill packages.

---

### 6. Channels System (`channels/`)

Multi-protocol message routing with session management.

- **`BaseChannel[T]`** — Abstract base: `start()`, `is_mentioned()`, `get_session_prompt()`, `process_output()`.

- **`TelegramChannel`** — Processes all private messages; group messages only when mentioned by keyword/username or replied to. Access control via `BUB_TELEGRAM_ALLOW_FROM` / `BUB_TELEGRAM_ALLOW_CHATS`. Markdown rendering via `telegramify_markdown`.

- **`DiscordChannel`** — Access control via user IDs and channel allow-lists. Configurable command prefix (default `!`). Supports embeds and threads.

- **`ChannelManager`** — Orchestrates all registered channels. Extensible via `BUB_HOOKS_MODULE`.

- **`SessionRunner`** — Per-session debouncing: batches rapid messages (1s debounce), waits for follow-ups if mentioned (10s delay), ignores messages outside active conversation window (60s). Commands (`,` prefix) execute immediately.

**Message flow:**
```
Telegram/Discord Message
  → ChannelManager → is_mentioned()? → SessionRunner (debounce/batch)
    → AppRuntime.handle_input()
      → Channel.process_output() → send response
```

---

### 7. CLI (`cli/`)

User-facing interfaces.

- **`chat`** (default) — Interactive REPL via `prompt_toolkit`. Features: tool name completion, Ctrl-X mode toggle (agent/shell), tape info display, history persistence per workspace.

- **`run`** — Single message execution for scripting.

- **`message`** — Start channel runtimes (Telegram, Discord).

- **`idle`** — Run scheduler only (autonomous execution of scheduled tasks).

---

### 8. Configuration (`config/`)

Pydantic `BaseSettings` with `BUB_` env prefix. Key settings:

| Category | Variables |
|----------|-----------|
| Model | `BUB_MODEL`, `BUB_API_KEY`, `BUB_MAX_TOKENS` (1024), `BUB_MAX_STEPS` (20) |
| Tape | `BUB_TAPE_NAME` ("bub"), `BUB_HOME` (~/.bub) |
| Telegram | `BUB_TELEGRAM_ENABLED`, `_TOKEN`, `_ALLOW_FROM`, `_ALLOW_CHATS` |
| Discord | `BUB_DISCORD_ENABLED`, `_TOKEN`, `_ALLOW_FROM`, `_ALLOW_CHANNELS` |
| Hooks | `BUB_HOOKS_MODULE` (import path for custom channel setup) |

---

### 9. Integrations (`integrations/`)

- **`build_llm()`** — Creates Republic LLM client. Supports multiple vendors: openrouter, azure, ollama, anthropic. Special handling for Azure API versions.

- **`build_tape_store()`** — Creates `FileTapeStore` pointing to workspace.

- **`read_workspace_agents_prompt()`** — Reads optional `AGENTS.md` for additional system context.

---

## Key Design Patterns

1. **Tape-first determinism** — All I/O captured to append-only JSONL; every interaction is replay-safe and auditable.

2. **Dual-mode routing** — Same command parsing logic for both user and assistant output ensures consistency.

3. **Progressive tool expansion** — Minimal tool summaries in prompts; full schemas only when hinted or used. Saves tokens.

4. **Skill hints (`$name`)** — Lazy activation of detailed skill docs avoids prompt bloat.

5. **Session isolation** — Forked tapes per user input prevent cross-session state leakage.

6. **Command wrapping** — Failed commands provide explicit `<command>` context to the model instead of silently failing.

7. **Decorator-based registry** — Clean separation of tool definitions and execution.

8. **Async-first** — asyncio throughout, especially for channel concurrency.

9. **Extensibility** — Hooks module for custom channels; skill discovery from multiple directories; allowed_tools/allowed_skills filters.

---

## Testing Structure

Tests use pytest + pytest-asyncio. Key test files:

- `test_command_detector.py` — Command detection logic
- `test_channels.py` — Channel base functionality
- `test_tape_*.py` — Tape service and store
- `test_model_runner.py` — Model loop and routing
- `test_tool_registry.py` — Tool registration and execution
- `test_skills_loader.py` — Skill discovery
- `test_tools_schedule.py` — Scheduled reminders
- `test_telegram_filter.py`, `test_discord_filter.py` — Channel filtering

Pattern: mock router/tape for unit tests, fake router implementations for scenario testing.
