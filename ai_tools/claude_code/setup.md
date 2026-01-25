# claude setup
focus on setup for sandbox env

```
# install claude
curl -fsSL https://claude.ai/install.sh | bash

# update .bashrc
export IS_SANDBOX=1
export PATH="$HOME/.local/bin:$PATH"
alias ccusage="BUN_BE_BUN=1 claude x ccusage"

# start claude
claude --dangerously-skip-permissions

# update model
/model [ops]

# install plugins
playwright
pyright-lsp
superpowers
claude-md-management
```