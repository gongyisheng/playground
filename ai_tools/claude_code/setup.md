# claude setup
focus on setup for sandbox env

```
# install claude
curl -fsSL https://claude.ai/install.sh | bash

# update .bashrc
export IS_SANDBOX=1
export PATH="$HOME/.local/bin:$PATH"
alias yolo="claude --dangerously-skip-permissions"

source ~/.bashrc

# copy CLAUDE.md
mkdir -p ~/.claude && curl -fsSL https://raw.githubusercontent.com/gongyisheng/playground/dev/ai_tools/claude_code/CLAUDE.md -o ~/.claude/CLAUDE.md

# start claude
yolo
```