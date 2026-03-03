# Pi Monorepo — High-Level Design

> Tools for building AI agents and managing LLM deployments.
> Author: Mario Zechner (@badlogic). Repo: https://github.com/badlogic/pi-mono

## Overview

Pi is a TypeScript monorepo organized as **7 npm packages** with a clear layered architecture. The bottom layers provide LLM abstraction and agent primitives; upper layers compose them into concrete products (coding CLI, Slack bot, web chat UI, GPU pod manager).

```
┌──────────────────────────────────────────────────────┐
│              Applications / Products                  │
│  ┌──────────────┐  ┌─────────┐  ┌────────────────┐  │
│  │ coding-agent │  │   mom   │  │    web-ui      │  │
│  │  (CLI)       │  │ (Slack) │  │ (Web Components│  │
│  └──────┬───────┘  └────┬────┘  └───────┬────────┘  │
│         │               │               │            │
│  ┌──────┴───────────────┴───────────────┘            │
│  │                                                   │
│  ▼                                                   │
│  ┌─────────────────────────────────────────────────┐ │
│  │                agent (runtime)                  │ │
│  │   State management, tool orchestration, events  │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                            │
│  ┌──────────────────────▼──────────────────────────┐ │
│  │                  ai (LLM API)                   │ │
│  │  Unified multi-provider streaming & tool calling│ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌──────────┐  ┌──────────┐                          │
│  │   tui    │  │   pods   │  (standalone utilities)  │
│  └──────────┘  └──────────┘                          │
└──────────────────────────────────────────────────────┘
```

**Build system**: npm workspaces, TypeScript (tsgo compiler), Biome formatter/linter, Vitest tests, lockstep versioning across all packages. Cross-platform binaries via `bun build --compile`.

---

## 1. `@mariozechner/pi-ai` — Unified LLM API

**Path**: `packages/ai`
**Purpose**: Provider-agnostic abstraction over 20+ LLM providers (500+ models) with streaming, tool calling, cost tracking, and cross-provider message transformation.

### Core Abstractions

| Concept | Description |
|---------|-------------|
| **Model** | Typed descriptor: id, provider, API type, cost, context window, capabilities (reasoning, images) |
| **Message** | Union of `UserMessage`, `AssistantMessage`, `ToolResultMessage`. Assistant content is a list of `TextContent`, `ThinkingContent`, or `ToolCall` |
| **Tool** | Name + description + TypeBox parameter schema. Validated at runtime via AJV |
| **Context** | Serializable conversation: system prompt + messages + tools |
| **EventStream\<T, R\>** | Async iterable that also resolves to a final result — enables both streaming UI and await-for-complete patterns |

### Provider Architecture

Each provider registers an `ApiProvider` with the global registry:

```
registerApiProvider({ api: "anthropic-messages", stream: ..., streamSimple: ... })
registerApiProvider({ api: "openai-completions", stream: ..., streamSimple: ... })
registerApiProvider({ api: "google-generative-ai", ... })
...
```

Built-in providers: OpenAI, Anthropic, Google (Generative AI + Vertex), AWS Bedrock, Mistral, Groq, Cerebras, xAI, OpenAI Codex, Google Gemini CLI, plus any OpenAI-compatible endpoint.

Streaming events are normalized to a common type:

```
start → text_delta* → thinking_delta* → toolcall_end* → done|error
```

### Key Design Patterns

- **Plugin registry**: Providers can be registered/unregistered dynamically (supports extensions and testing)
- **Cross-provider message transformation**: Normalizes tool call IDs, converts thinking blocks, injects synthetic tool results for orphaned calls
- **Generated model catalog**: `models.generated.ts` auto-generated from external source with full cost/capability metadata
- **OAuth provider registry**: Extensible OAuth for Anthropic, GitHub Copilot, Google, OpenAI, etc.
- **Environment-aware key resolution**: Handles special cases (AWS credential chain, GH_TOKEN precedence, ADC for Vertex)

---

## 2. `@mariozechner/pi-agent-core` — Agent Runtime

**Path**: `packages/agent`
**Purpose**: Stateful agent framework with tool execution, streaming events, and message management. Thin layer on top of pi-ai.

### Core Abstractions

| Concept | Description |
|---------|-------------|
| **Agent** | High-level stateful wrapper: manages messages, model, tools, streaming state. Methods: `prompt()`, `continue()`, `abort()`, `steer()`, `followUp()` |
| **agentLoop()** | Low-level function: the inner while-loop that calls the LLM, executes tools, checks for steering interrupts, and repeats |
| **AgentMessage** | Superset of LLM Message — includes custom app messages (UI notifications, artifacts) that are filtered out before LLM calls via `convertToLlm()` |
| **AgentTool** | Extends Tool with `label` and `execute()`. Supports streaming updates and error reporting |
| **AgentEvent** | 9 event types: agent start/end, turn start/end, message start/update/end, tool execution start/update/end |

### Agent Loop Flow

```
prompt(messages)
  └─ agentLoop:
       while (hasToolCalls || pendingMessages):
         transformContext()   // optional: prune, inject, summarize
         convertToLlm()      // filter out custom messages
         LLM call (streaming)
         if toolCalls:
           for each tool:
             execute tool
             check steering queue (user interrupt → skip remaining tools)
         emit events
```

### Key Design Patterns

- **Message filtering**: Custom app messages (artifacts, UI events) live in conversation history but are automatically stripped before LLM calls
- **Steering & follow-up queues**: Real-time user interruption during tool execution; deferred messages after agent stops
- **Transport abstraction**: Pluggable `streamFn` + `streamProxy()` for browser/proxied backends
- **Composition over inheritance**: Agent composes `agentLoop()` functions; low-level API available for advanced use
- **Declaration merging extensibility**: TypeScript `CustomAgentMessages` interface for adding custom message types

---

## 3. `@mariozechner/pi-coding-agent` — Interactive Coding CLI

**Path**: `packages/coding-agent`
**Purpose**: The main product — a terminal-based coding agent (the `pi` command) with session management, extensibility, and multiple run modes.

### Architecture

```
CLI (cli.ts / main.ts)
  │
  ├─ Interactive Mode ──→ TUI (pi-tui)
  ├─ Print Mode ────────→ stdout (one-shot)
  ├─ JSON Mode ─────────→ streaming JSON
  ├─ RPC Mode ──────────→ stdin/stdout protocol
  └─ Export ────────────→ HTML
  │
  ▼
AgentSession (agent-session.ts, 99KB)
  ├─ Wraps Agent (pi-agent-core)
  ├─ Extension runner (event dispatch, tool wrapping)
  ├─ Session persistence (JSONL with tree structure)
  ├─ Context compaction (automatic/manual summarization)
  ├─ Model registry (provider discovery, API key resolution)
  └─ Bash executor
```

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Session** | Persistent conversation stored as JSONL. Tree structure (id/parentId) enables in-place branching without new files |
| **Session entries** | Messages, compaction summaries, branch summaries, model changes, custom entries, labels |
| **Extensions** | Plugins that register tools, hook into events, add commands/shortcuts/UI. Built via TypeScript/JS modules |
| **Compaction** | Automatic context summarization when approaching token limits. Full history preserved; `/tree` allows revisiting any point |
| **Resources** | Extensions, skills, prompt templates, themes, context files — discovered from project `.pi/`, user `~/.pi/agent/`, and installed packages |

### Built-in Tools

7 tools: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`. Each has a factory function for custom cwd/options.

### Extension System

Three APIs: tool registration, event hooks (agent lifecycle, tool calls, session events, input events), and UI/command registration. Enables building MCP integration, git checkpointing, permission gates, custom editors, etc. — all as extensions rather than core features.

### Key Design Patterns

- **Mode as thin wrapper**: All modes share AgentSession; mode only adds I/O semantics
- **Event-driven architecture**: Extensions subscribe to granular events
- **Wrapper/hook pattern**: Built-in tools wrapped by extension hooks before LLM invocation
- **Lazy resource loading**: Hot-reload (`/reload`) without restart
- **Credential abstraction**: AuthStorage + ModelRegistry with `${ENV_VAR}` substitution

---

## 4. `@mariozechner/pi-tui` — Terminal UI Library

**Path**: `packages/tui`
**Purpose**: Minimal, flicker-free TUI framework with differential rendering, ~4,250 lines of TypeScript.

### Core Abstractions

```typescript
interface Component {
  render(width: number): string[]   // returns lines, max width enforced
  handleInput?(data: string): void  // keyboard input
  invalidate?(): void               // clear cached state
}

// TUI extends Container — manages rendering, overlays, focus, input dispatch
```

### Rendering Strategy

Three modes selected automatically:
1. **First render**: Output all lines (assumes clean terminal)
2. **Width changed / change above viewport**: Full clear + re-render
3. **Normal update**: Cursor to first changed line, clear-to-end, render only delta

All wrapped in **synchronized output** (`CSI 2026`) for atomic, flicker-free display.

### Key Components

| Component | Description |
|-----------|-------------|
| **Editor** | Full multi-line editor with autocomplete, paste, undo/redo (64KB) |
| **Input** | Single-line input with kill-ring (Emacs-style), undo |
| **Markdown** | Renderer with syntax highlighting and theming |
| **SelectList** | Keyboard-navigable selection |
| **SettingsList** | Settings panel with value cycling and submenus |
| **Image** | Inline images via Kitty/iTerm2 graphics protocols |
| **Overlay** | Stack-based modal system with configurable positioning |

### Key Design Patterns

- **Width-aware text**: `visibleWidth()` accounts for ANSI codes, emoji, CJK characters via `Intl.Segmenter` + East Asian width tables
- **Kitty keyboard protocol**: Modern terminal input with key release events, modifiers
- **StdinBuffer**: Accumulates partial escape sequences across multiple stdin events
- **IME support**: `CURSOR_MARKER` (zero-width APC sequence) at cursor position for input method candidate windows
- **Debug diagnostics**: Width overflow crashes write detailed debug info to `~/.pi/agent/pi-crash.log`

---

## 5. `@mariozechner/pi-web-ui` — Web Components for AI Chat

**Path**: `packages/web-ui`
**Purpose**: Reusable Lit-based web components for building AI chat interfaces in the browser.

### Architecture

```
ChatPanel (responsive split-view, 800px breakpoint)
├── AgentInterface (left: chat)
│   ├── MessageList (stable messages)
│   ├── StreamingMessageContainer (live streaming)
│   └── MessageEditor (input + attachments + model selector)
└── ArtifactsPanel (right: generated content)
    ├── HtmlArtifact (SandboxedIframe)
    ├── SvgArtifact, MarkdownArtifact, TextArtifact
    ├── ImageArtifact, PdfArtifact, DocxArtifact, ExcelArtifact
    └── Artifact CRUD tool
```

### Storage Layer

IndexedDB-backed with transaction support:
- **SessionsStore**: Chat sessions + metadata (dual-store for fast listing)
- **ProviderKeysStore**: API keys by provider
- **SettingsStore**: Key-value settings
- **CustomProvidersStore**: User-defined Ollama/LM Studio/vLLM endpoints

### Sandbox System

JavaScript REPL and HTML artifacts execute in isolated iframes via `postMessage` bridge:

```
Main Window ←→ RuntimeMessageRouter ←→ Iframe
                 ├─ ConsoleProvider (captures logs)
                 ├─ ArtifactsProvider (read/write artifacts)
                 ├─ AttachmentsProvider (access files)
                 └─ FileDownloadProvider (return files)
```

### Key Design Patterns

- **Light DOM web components**: `createRenderRoot() { return this }` for shared Tailwind CSS styling
- **Registry patterns**: Pluggable message renderers and tool result renderers
- **Dual-store sessions**: Lightweight metadata index for fast listing without loading full history
- **Attachment processing**: Polymorphic text extraction (PDF, DOCX, XLSX) for document search/analysis
- **CORS proxy handling**: Configurable proxy for browser-based API calls

---

## 6. `@mariozechner/pi-mom` — Slack Bot

**Path**: `packages/mom`
**Purpose**: "Master Of Mischief" — a Slack bot that delegates messages to the pi coding agent with full bash/file access.

### Architecture

```
Slack (Socket Mode)
  ↓ @mention or DM
Message Queue (per-channel, sequential)
  ↓
Agent Runner (cached per channel)
  ├─ System prompt (memory + skills + channel/user mappings)
  ├─ Session context (synced from log.jsonl → context.jsonl)
  └─ Tool execution
  ↓
Executor (host bash or Docker container)
  ├─ bash, read, write, edit, attach tools
  └─ Results posted to Slack threads
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Per-channel isolation** | Each channel gets its own directory, runner instance, memory, and skills |
| **Dual persistence** | `log.jsonl` (source of truth for all messages) + `context.jsonl` (LLM context, synced from log) |
| **Events system** | File-based event queue with immediate, one-shot (scheduled), and periodic (cron) triggers |
| **Memory & Skills** | `MEMORY.md` for persistent context; custom CLI skills for recurring tasks. Global + channel-specific hierarchy |
| **Sandbox modes** | Host executor (direct bash, dangerous) or Docker executor (isolated, recommended) |

### Key Design Patterns

- **Self-managing workspace**: Mom installs her own tools, configures credentials, creates custom skills
- **Streaming Slack updates**: Accumulated message building with " ..." working indicator
- **Thread-based tool details**: Tool inputs/outputs posted to threads, keeping the main conversation clean
- **Silent response support**: `[SILENT]` marker for background processing
- **Auto-compaction**: When context exceeds window size, old messages are summarized

---

## 7. `@mariozechner/pi-pods` — GPU Pod Manager

**Path**: `packages/pods`
**Purpose**: CLI for provisioning and managing vLLM deployments on remote GPU pods.

### Workflow

```bash
# 1. Setup pod (one-time)
pi pods setup dc1 "ssh root@1.2.3.4" --mount "..." --vllm 0.10.0
  # → Installs CUDA, Python, vLLM, mounts storage

# 2. Deploy model
pi start Qwen/Qwen2.5-Coder-32B-Instruct --name qwen
  # → Looks up config in models.json
  # → Allocates GPUs (round-robin, least-used first)
  # → Starts vLLM server on next available port
  # → Monitors startup (success/OOM/failure detection)

# 3. Use model
pi agent qwen -i   # Interactive chat
pi list             # Status of all models
pi logs qwen        # Monitor
pi stop qwen        # Stop
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Model configs** | `models.json` with predefined vLLM args per model, with hardware-specific GPU configurations (1/2/8/16 GPU setups) |
| **Smart GPU allocation** | Round-robin assignment to least-used GPUs; multiple models can share a pod |
| **SSH-based execution** | Commands via SSH with keepalive; `setsid` detaches processes from SSH session |
| **Config persistence** | `~/.pi/pods.json` tracks pods, GPUs, running models (port, GPU IDs, PID) |
| **Startup monitoring** | Streams logs watching for success/OOM/failure markers with contextual suggestions |

Supported models: Qwen family (2.5-Coder, 3-Coder), OpenAI GPT-OSS, GLM-4.5, Kimi-K2. All expose OpenAI-compatible `/v1/chat/completions` endpoints.

---

## Cross-Cutting Concerns

### Dependency Graph

```
coding-agent ──→ agent ──→ ai
     │              │
     ├──→ tui       │
     │              │
mom ─┼──→ agent ────┘
     │
web-ui ──→ ai
     │
pods ──→ agent
```

### Conventions

- **Language**: TypeScript (strict mode), ES2022, ESM only
- **Formatting**: Biome — tabs (3-space width), 120 char lines
- **Testing**: Vitest with 30s timeout for API calls
- **Versioning**: Lockstep across all packages (patch = fixes/features, minor = breaking)
- **Releases**: Automated via `scripts/release.mjs` — bumps versions, updates CHANGELOGs, tags, publishes to npm
- **Binary distribution**: `bun build --compile` for 5 platforms (darwin-arm64/x64, linux-x64/arm64, windows-x64)
- **Node requirement**: >=20.0.0
