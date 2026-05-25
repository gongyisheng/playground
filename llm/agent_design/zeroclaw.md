# ZeroClaw - High Level Design

> Zero overhead. Zero compromise. 100% Rust. 100% Agnostic.

ZeroClaw is a high-performance, minimal-footprint Rust agent runtime optimized for edge devices and constrained environments. Single binary deployment (<5MB RAM, <10ms startup), trait-driven architecture with swappable providers/channels/tools/memory, and secure-by-default runtime.

## Workspace Structure

```
zeroclaw/
├── src/                     # main binary + library (monolith)
├── crates/
│   ├── zeroclaw-types/      # foundational shared types (scaffolding)
│   ├── zeroclaw-core/       # core contracts and boundaries (scaffolding)
│   └── robot-kit/           # standalone robotics toolkit
├── clients/
│   ├── android/             # Kotlin/Compose native app
│   └── android-bridge/      # Rust → Android FFI via UniFFI
├── web/                     # React 19 management dashboard
├── python/                  # LangGraph-based Python agent framework
├── firmware/                # microcontroller firmware (ESP32, Arduino, STM32)
├── extensions/              # WASM plugin examples
├── templates/               # starter templates (Go, Python, Rust, TS)
├── wit/                     # WebAssembly Interface Type definitions
├── docs/                    # comprehensive documentation (112 files)
├── site/                    # React-based docs site
└── examples/                # trait implementation examples
```

## Core Modules (src/)

### Agent (`src/agent/`)

Orchestration loop and session management. Contains the main agent loop that receives messages, classifies intent, dispatches to tools/providers, and manages conversation sessions.

Key components: `Agent`, `AgentBuilder`, classifier, dispatcher, session manager, team orchestration (multi-agent), quota-aware execution, research mode.

### Providers (`src/providers/`)

Multi-backend LLM inference with factory pattern. Supports 15+ providers:

- **Cloud**: OpenAI, Anthropic, Gemini, Bedrock (AWS), OpenRouter
- **Local**: Ollama
- **Regional**: GLM (Zhipu), Moonshot (Kimi), Qwen (Alibaba), SiliconFlow, StepFun, Minimax
- **IDE**: Copilot, Cursor

Features resilient multi-provider fallback chains, model routing across providers, quota management, cost tracking, health checks, and backoff strategies.

### Channels (`src/channels/`)

Multi-platform messaging integrations behind a uniform `Channel` trait:

- **Chat**: Telegram, Discord, Slack, WhatsApp, Matrix (E2EE), iMessage
- **Dev**: GitHub
- **Comms**: Email, IRC, MQTT, Nostr
- **Enterprise**: Lark (Feishu), DingTalk
- **Custom**: ClawdTalk, BlueBubbles

Features per-sender conversation history, concurrent message processing, exponential-backoff reconnection, typing indicators.

### Tools (`src/tools/`)

70+ tool implementations providing agent capabilities:

| Category | Tools |
|----------|-------|
| File I/O | file_read, file_write, file_edit, glob_search |
| Web | web_fetch, web_search, browser automation (fantoccini) |
| Code | git_operations, shell execution |
| Documents | pdf_read, docx_read, xlsx_read, pptx_read |
| Hardware | GPIO control, sensor reading |
| Coordination | cron scheduling, delegation, subagent spawning |
| Memory | memory_store, memory_recall, memory_forget |
| Integrations | Composio (200+ external integrations), MCP client, Firecrawl |
| Planning | task_plan, model_routing_config |

### Memory (`src/memory/`)

Pluggable storage backends for conversation history, long-term memory, and vector search:

- **Local**: SQLite, Markdown (file-based)
- **Remote**: PostgreSQL, Qdrant (vector DB)
- **Hybrid**: SQLite + Qdrant, LucidMemory, CortexMem

Features memory classification (core/daily/conversation/custom), embeddings generation, vector search, memory decay policies, hygiene/privacy checks, response caching.

### Security (`src/security/`)

Deny-by-default access control:

- **Identity**: OTP-based pairing
- **Authorization**: RBAC (role-based access control)
- **Isolation**: Landlock (Linux), Firejail, Bubblewrap, Docker sandboxing
- **Secrets**: Encrypted store (ChaCha20Poly1305)
- **Detection**: Syscall anomaly detection, prompt guard
- **Auditing**: Action logging and audit trails
- **Safety**: Emergency stop (E-Stop) at multiple levels

### Gateway (`src/gateway/`)

HTTP/WebSocket server for remote access. RESTful API with OpenAI-compatible mode, WebSocket for real-time updates, SSE, static file serving for the web dashboard. Runs on port 5555.

### Peripherals (`src/peripherals/`)

Hardware board integration. Supports STM32 Nucleo, Arduino, ESP32, Raspberry Pi GPIO. Device enumeration via probe-rs, serial communication with JSON protocol.

### Other Modules

- `auth/` - OAuth, API key authentication
- `daemon/` - system daemon management
- `doctor/` - health checks and diagnostics
- `coordination/` - multi-agent coordination
- `goals/` - goal/objective management
- `rag/` - retrieval-augmented generation
- `runtime/` - runtime adapters
- `plugins/` - WASM plugin system
- `skills/` - skill factory and management
- `cron/` - scheduled task execution
- `hooks/` - event lifecycle hooks
- `tunnel/` - reverse tunneling
- `multimodal/` - text + image handling
- `observability/` - OpenTelemetry, Prometheus, structured logging
- `config/` - TOML-based configuration with JSON schema generation

## Standalone Crates

### zeroclaw-robot-kit (`crates/robot-kit/`)

Standalone robotics toolkit for AI-powered autonomous robots. Provides:

- `DriveTool` - motor control (ROS2, serial, GPIO, mock backends)
- `LookTool` - camera capture + vision model description (Ollama)
- `ListenTool` - speech-to-text via Whisper.cpp
- `SpeakTool` - text-to-speech via Piper TTS
- `SenseTool` - LIDAR, motion sensors, ultrasonic distance
- `EmoteTool` - LED matrix expressions and sounds
- `SafetyMonitor` - independent safety supervisor running in parallel

Key design: safety runs as an independent background task that can override AI decisions. AI can REQUEST movement, but SafetyMonitor ALLOWS it, preventing collisions even if the LLM hallucinates.

### zeroclaw-types / zeroclaw-core

Scaffolding crates for staged workspace modularization. Currently minimal - types holds shared type definitions, core holds trait contracts. The monolith in `src/` is being gradually extracted into these.

## Clients

### Web Dashboard (`web/`)

React 19 + Tailwind CSS 4 + Vite SPA. Pages: Dashboard, AgentChat, Tools, Cron, Integrations, Memory, Devices, Config, Cost tracking, Logs, Doctor diagnostics. Communicates with core via REST/WebSocket/SSE at localhost:5555. Has i18n support.

### Android (`clients/android/` + `clients/android-bridge/`)

Native Kotlin/Compose app with Rust core compiled via UniFFI. The bridge exposes `ZeroClawController`, `ChatMessage`, `AgentStatus` to Kotlin. Uses WorkManager for background tasks, Android Keystore for secrets. Optimized for <3MB per ABI.

### Python Framework (`python/`)

LangGraph-based agent framework (`zeroclaw-tools` package). Provides `ZeroclawAgent` class, `@tool` decorator, built-in tools (shell, file I/O, web search, HTTP, memory), and Discord bot integration. Works with any OpenAI-compatible API. Designed to handle tool-calling inconsistencies in providers like GLM-5.

## Firmware

JSON-over-serial protocol bridges for microcontrollers. Host ZeroClaw sends GPIO commands; firmware executes and responds.

| Board | Language | Framework |
|-------|----------|-----------|
| ESP32 | Rust | ESP-IDF |
| Arduino Uno | C | Arduino IDE |
| STM32 Nucleo | Rust | Embassy (async) |

## Plugin System (WASM)

Extensibility via WebAssembly component model. Three plugin interfaces defined in WIT:

1. **Tools** (`wit/zeroclaw/tools/v1/`) - `list-tools()` + `execute-tool(name, args)` → custom agent capabilities
2. **Providers** (`wit/zeroclaw/providers/v1/`) - `chat(request)` → custom LLM backends
3. **Hooks** (`wit/zeroclaw/hooks/v1/`) - lifecycle events (gateway start/stop, session start/end, compaction) → can continue or cancel

Templates in `templates/` provide starters in Go (TinyGo → WASI), Python (componentize-py), Rust, and TypeScript.

## Key Architectural Patterns

1. **Trait-driven extensibility** - Provider, Channel, Tool, Memory, Peripheral traits enable pluggable implementations
2. **Factory pattern** - string-based key lookup for all subsystems ("openai", "discord", "shell")
3. **Security by default** - deny-by-default policies, encrypted secrets, sandbox isolation, audit logging
4. **Resilience** - multi-provider fallback chains, exponential backoff, health checks, quota awareness
5. **Binary size optimization** - opt-level=z, LTO=fat, strip=true → ~8.8MB release binary

## CLI Commands

```
zeroclaw chat         # direct chat with agent
zeroclaw gateway      # start web dashboard server
zeroclaw channel      # manage messaging platforms
zeroclaw provider     # configure AI model providers
zeroclaw tool         # tool management
zeroclaw memory       # memory operations
zeroclaw cron         # scheduled tasks
zeroclaw peripheral   # hardware setup
zeroclaw skill        # skill install/management
zeroclaw migrate      # OpenClaw data import
zeroclaw service      # system daemon management
zeroclaw doctor       # health diagnostics
```

## Data Flow

```
User Message
    │
    ▼
Channel (Telegram/Discord/Web/...) ──► Agent Loop
    │                                      │
    │                              ┌───────┼───────┐
    │                              ▼       ▼       ▼
    │                          Classify  Memory  Provider
    │                              │     Recall   (LLM)
    │                              ▼               │
    │                          Dispatch ◄──────────┘
    │                              │
    │                    ┌─────────┼─────────┐
    │                    ▼         ▼         ▼
    │                  Tools    Plugins   Peripherals
    │                 (70+)    (WASM)    (GPIO/Serial)
    │                    │         │         │
    │                    └─────────┼─────────┘
    │                              ▼
    │                        Memory Store
    │                              │
    ▼                              ▼
Channel Response ◄──────── Formatted Reply
```
