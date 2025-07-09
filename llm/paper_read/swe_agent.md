# SWE Agent
title: SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
url: https://arxiv.org/pdf/2405.15793?

## Key takeaways
- LLM agent need a new interface. GUI is human friendly but bad for LLM
- Best ACI would be command line tool + file + well designed tools
- The goal is to provide accurate and concise context for agent with accurate feedbacks.

## Design
- shell commands
- file
- well design tools (search, file viewer, file editor)

### Principle
- Actions should be simple and easy to understand for agents.
- Actions should be compact and efficient.
- Environment feedback should be informative but concise.
- Guardrails mitigate error propagation and hasten recovery.

### Tools
1. search
    - search_dir: locate string in a file for target dir (only 50 result, if not retry)
    - search_file: locate string in a file for target file (only 50 result, if not retry)
    - find_file: search for a specific file by its name in repo
2. file viewer
    - open: open a file with path
    - scroll up / scroll down: manage a context window of 100 lines, provide number of lines above and below
    - goto: view a specific line
3. file editor
    - create: create a new file
    - edit: replace line m to n with new code block, support linting
4. context management
    - each round generate a thought and an action
    - error response trigger a retry
    - tool response is summarized or limited
    - last 5 rounds of conversation as context

### Other Findings
1. Even grep/find/cat/ls/cd can be inefficient, should be well designed for LLM
2. Compact, efficient file editing is critical to performance.
3. A failure mode is repeatedly edit the same code snippet, add linting (guardrails) can reduce its occurance
4. LLM editting is likely run into error (only 90% succ rate each round)
5. Agent succeed quickly, fail slowly
  