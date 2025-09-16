# AI tool configurations
## Install 
linux:
```
mkdir -p ~/.npm_global
npm config set prefix '~/.npm_global'

edit ~/.bashrc
export PATH=$HOME/.npm_global/bin:$PATH
source ~/.bashrc

npm install -g @openai/codex
```

macos:
```
brew install codex
```
## Codex
## Claude Code
### Commands
```
mkdir -p .claude/commands
echo "Analyze this code for performance issues and suggest optimizations" > .claude/commands/optimize.md
```
### MCP Server
```
Add MCP server:
claude mcp add <server> --scope user

Remove MCP server:
claude mcp remove <server>

Add Atlassian MCP server: (https://github.com/sooperset/mcp-atlassian)
docker pull ghcr.io/sooperset/mcp-atlassian:latest
claude mcp add mcp-atlassian \
  --scope user \
  --env CONFLUENCE_URL=https://company.atlassian.net/wiki \
  --env CONFLUENCE_USERNAME=email@company.com \
  --env CONFLUENCE_API_TOKEN= \
  --env JIRA_URL=https://company.atlassian.net \
  --env JIRA_USERNAME=email@company.com \
  --env JIRA_API_TOKEN= \
  -- docker run -i --rm \
  -e CONFLUENCE_URL \
  -e CONFLUENCE_USERNAME \
  -e CONFLUENCE_API_TOKEN \
  -e JIRA_URL \
  -e JIRA_USERNAME \
  -e JIRA_API_TOKEN \
  ghcr.io/sooperset/mcp-atlassian:latest

Add Github MCP server: (https://github.com/github/github-mcp-server)
docker pull ghcr.io/github/github-mcp-server:latest
claude mcp add mcp-github \
  --scope user \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=
  -- docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN \
  ghcr.io/github/github-mcp-server:latest

Add n8n MCP server: (https://github.com/czlonkowski/n8n-mcp)
docker pull ghcr.io/czlonkowski/n8n-mcp:latest
claude mcp add mcp-n8n \
  --scope user \
  -- docker run -i --rm \
  -e MCP_MODE=stdio \
  -e LOG_LEVEL=error \
  -e DISABLE_CONSOLE_OUTPUT=true \
  -e N8N_API_URL= \
  -e N8N_API_KEY= \
  ghcr.io/czlonkowski/n8n-mcp:latest 
```