# Nanobot - Architecture & Design

Ultra-lightweight personal AI assistant framework (~4,000 lines Python). 99% smaller than Clawdbot while delivering core agent functionality: multi-channel chat, persistent memory, tool execution, MCP integration, and scheduled tasks.

## System Architecture

```
User (Telegram/Discord/WhatsApp/Email/...)
    │
    ▼
┌─────────────────────────┐
│   Channels Layer        │  9 platform integrations
│   (BaseChannel impl)    │  each: on_message() + send()
└────────┬────────────────┘
         │ InboundMessage / OutboundMessage
         ▼
┌─────────────────────────┐
│   Message Bus           │  async FIFO queues
│   (inbound + outbound)  │  decouples channels ↔ agent
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│   AgentLoop                                 │
│   ┌───────────────┐  ┌──────────────────┐   │
│   │ ContextBuilder│  │ SessionManager   │   │
│   │ (prompt assy) │  │ (JSONL history)  │   │
│   └───────────────┘  └──────────────────┘   │
│   ┌───────────────┐  ┌──────────────────┐   │
│   │ MemoryStore   │  │ ToolRegistry     │   │
│   │ (MEMORY.md +  │  │ (32+ tools +     │   │
│   │  HISTORY.md)  │  │  MCP servers)    │   │
│   └───────────────┘  └──────────────────┘   │
│   ┌───────────────┐                         │
│   │ LLMProvider   │                         │
│   │ (multi-model) │                         │
│   └───────────────┘                         │
└─────────────────────────────────────────────┘
```

## Data Flow

```
1. User sends message on channel (e.g. Telegram)
2. Channel.on_message() → InboundMessage → bus.inbound queue
3. AgentLoop consumes message, loads/creates session (key = "channel:chat_id")
4. ContextBuilder assembles system prompt:
   identity + bootstrap files + long-term memory + skills
5. Messages = [system, ...history, user_message] → LLMProvider.chat()
6. If LLM returns tool_calls:
     execute each tool → add results to context → loop back to LLM
7. Final response → session history (append-only)
8. Consolidation check: if messages > memory_window → summarize old → files
9. OutboundMessage → bus.outbound → ChannelManager → channel.send()
```

---

## Module Breakdown

### 1. Agent (`agent/`)

The core processing engine. 16 files.

**`loop.py` — AgentLoop**
- Main event loop: consume inbound messages, call LLM iteratively with tools, produce outbound responses.
- Iterative tool calling: LLM response may contain tool calls → execute → feed results back → repeat until final text response.
- Manages session state and triggers memory consolidation when history grows too long.

**`context.py` — ContextBuilder**
- Assembles the system prompt from multiple sources:
  - Identity files (AGENTS.md, SOUL.md, USER.md, TOOLS.md, IDENTITY.md)
  - Long-term memory (MEMORY.md)
  - Active skills (full content for `always` skills, summary for others)
  - Runtime metadata (time, channel, chat_id)
- Injects untrusted runtime context before each user message.

**`memory.py` — MemoryStore**
- Two-layer persistence:
  - **MEMORY.md**: long-term facts (user-managed, loaded into every prompt)
  - **HISTORY.md**: grep-searchable conversation log (auto-appended)
- Consolidation: old messages → LLM summarization → append to memory files.
- Triggered when session messages exceed `memory_window` (default 100).

**`skills.py` — Skill Loader**
- Discovers SKILL.md files from workspace and built-in directories.
- YAML frontmatter metadata: `requires`, `description`, `always`, etc.
- Progressive loading: summary in context, full content loaded on-demand via `read_file` tool.
- Availability checking: verifies binary dependencies and env vars before enabling a skill.

**`subagent.py` — Subagent**
- Background task execution spawned via the `spawn` tool.
- Independent agent with limited tool subset (no message/spawn — prevents recursion).
- Max 15 iterations, reports completion to origin channel.

**`tools/` — Tool System (10 files)**

| File | Tools | Purpose |
|------|-------|---------|
| `base.py` | — | Abstract `Tool` class with JSON schema validation |
| `registry.py` | — | Dynamic tool registration + execution + error handling |
| `filesystem.py` | read_file, write_file, edit_file, list_dir | File operations with path traversal protection |
| `shell.py` | exec | Shell command execution with dangerous command blocking |
| `web.py` | web_search, web_fetch | Brave search API + HTTP fetch with proxy support |
| `message.py` | send | Route messages to external channels |
| `cron.py` | cron_add, cron_remove, cron_list, cron_run | Scheduled task management |
| `spawn.py` | spawn | Launch subagent for background work |
| `mcp.py` | (dynamic) | MCP server integration (stdio + HTTP transports) |

Security: JSON schema validation on all tool calls, 10KB output truncation, 60s shell timeout.

---

### 2. Bus (`bus/`)

Async message queue for decoupling channels from the agent. 3 files.

- **`queue.py`**: Two asyncio.Queue instances — `inbound` (channels → agent) and `outbound` (agent → channels). Simple FIFO, allows concurrent channel + agent operation.
- **`events.py`**: Message types — `InboundMessage` (channel, sender_id, chat_id, content, media, metadata) and `OutboundMessage` (channel, chat_id, content, reply_to, media).

The bus is the central spine — channels only know about the bus, the agent only knows about the bus. Neither knows about each other directly.

---

### 3. Channels (`channels/`)

Chat platform integrations. 13 files. All inherit `BaseChannel` and implement `on_message()` + `send()`.

| Channel | Transport | Notes |
|---------|-----------|-------|
| Telegram | Polling | Voice transcription support |
| Discord | WebSocket | Role-based access control |
| WhatsApp | Node.js bridge | QR code authentication |
| Feishu | WebSocket long-conn | Lark SDK, media support |
| Slack | Socket mode | Thread isolation |
| DingTalk | Stream mode | No public IP needed |
| Email | IMAP/SMTP polling | Periodic inbox check |
| QQ | botpy SDK | Sandbox support |
| Mochat | Socket.IO | Group policies |
| Matrix | Element | E2EE support |
| CLI | stdin/stdout | Interactive + single-message modes |

**`manager.py` — ChannelManager**: initializes enabled channels from config, routes outbound messages to the correct channel by name, handles concurrent startup/shutdown.

---

### 4. Config (`config/`)

Pydantic v2-based configuration system. 3 files.

**`schema.py`**: Root `Config` class with nested sections:
- `agents`: model, provider, temperature, max_tokens, memory_window, reasoning_effort
- `channels`: per-channel configs (token, allowFrom, etc.)
- `providers`: API keys/bases for 17+ LLM providers
- `tools`: web search keys, exec timeout, MCP server definitions, workspace restriction
- `gateway`: heartbeat interval, port

Provider matching logic: explicit prefix → keyword match → fallback. Alias generator handles camelCase ↔ snake_case.

**`loader.py`**: Loads from `~/.nanobot/config.json`, fallback to defaults, env var support for secrets.

---

### 5. Providers (`providers/`)

LLM abstraction layer. 7 files.

**`base.py`**: Abstract `LLMProvider` interface with `chat()` method. Returns `LLMResponse` (content, tool_calls, finish_reason, usage, reasoning_content).

**`registry.py`**: Single source of truth for 17+ providers — OpenRouter, Anthropic, OpenAI, DeepSeek, Groq, Gemini, Qwen, vLLM, GitHub Copilot, etc. Stores per-provider metadata: env vars, model prefixing rules, keywords, API base URLs. Supports gateway auto-detection by key prefix.

**Implementations:**
- `litellm_provider.py`: LiteLLM-based unified wrapper (handles prompt caching, reasoning effort, token counting)
- `custom_provider.py`: Direct OpenAI-compatible API for self-hosted endpoints (bypasses LiteLLM)
- `openai_codex_provider.py`: OAuth device flow for ChatGPT Plus/Pro accounts
- `transcription.py`: Voice-to-text via Groq Whisper

---

### 6. Session (`session/`)

Conversation state persistence. 2 files.

**`manager.py` — SessionManager**:
- One session per `channel:chat_id` pair.
- Storage: JSONL files in `workspace/sessions/`.
- Append-only in-session (never modify existing messages) — enables efficient LLM prompt caching.
- `get_history(max_messages)`: returns unconsolidated messages, aligned to user turn boundaries.
- `consolidate()`: moves old messages to memory files via LLM summarization.

---

### 7. Cron (`cron/`)

Scheduled task system. 3 files.

**`service.py` — CronService**:
- Storage: JSON file `~/.nanobot/data/cron/jobs.json`.
- Schedule types: **cron** expressions (e.g. `0 9 * * *`), **every** intervals (seconds), **at** one-time ISO timestamps.
- State tracking: next_run_at, enabled flag.
- Auto-reloads on external file modification.
- `on_job` callback triggers agent loop when job fires.

**`types.py`**: Data models — `CronSchedule`, `CronJob`, `CronStore`.

---

### 8. Heartbeat (`heartbeat/`)

Proactive periodic task execution. 2 files.

**`service.py` — HeartbeatService**:
- Two-phase execution:
  1. **Decision**: reads HEARTBEAT.md, asks LLM whether any tasks should run now
  2. **Execution**: if LLM says "run", triggers full agent loop for active tasks
- Default interval: 30 minutes (configurable).
- Use case: periodic checks (weather, emails, stock prices) without user trigger.

---

### 9. CLI (`cli/`)

User-facing command interface. 2 files. Built with Typer + Rich + prompt_toolkit.

**Key commands:**
- `nanobot onboard`: Initialize config + workspace
- `nanobot agent`: Chat mode (interactive or single message)
- `nanobot gateway`: Start full server (all channels + heartbeat + cron)
- `nanobot status`: Show config and API key status
- `nanobot cron add/list/remove/run`: Scheduled task management
- `nanobot channels login/status`: WhatsApp QR code + channel info

Interactive mode: multi-line paste, history, spinner during LLM generation, markdown rendering.

---

### 10. Utilities (`utils/`)

Common helpers. 2 files.

- Path management: `get_config_path()`, `get_workspace_path()`, `get_data_dir()`
- `safe_filename()`: sanitize filenames
- `sync_workspace_templates()`: copy default templates to workspace on init

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Append-only session messages** | Never modify messages in-session → enables LLM prompt caching (cache hit on prefix) |
| **Message bus decoupling** | Async queues separate channels from agent, allowing concurrent multi-channel operation |
| **Provider registry as single source of truth** | Prevents duplication, enables automatic env var + prefix management for 17+ providers |
| **Progressive skill loading** | Only skill summaries in system prompt, full content loaded on-demand → saves context window |
| **Consolidation over truncation** | Old messages → LLM-summarized → files, preserving searchable history instead of losing it |
| **Tool JSON schema validation** | Validates every tool call before execution, preventing invalid/malicious inputs |
| **Subagent tool restrictions** | No message/spawn in subagents → prevents recursive feedback loops |

## Extensibility

- **New channel**: inherit `BaseChannel`, implement `on_message()` + `send()`
- **New tool**: inherit `Tool`, implement `name`, `description`, `parameters`, `execute()`
- **New provider**: add entry to `PROVIDERS` registry + config field
- **New skill**: create `workspace/skills/skillname/SKILL.md` with YAML frontmatter
- **MCP servers**: add config entry in `tools.mcpServers`
- **Custom bootstrap**: place AGENTS.md / SOUL.md / USER.md / TOOLS.md in workspace
