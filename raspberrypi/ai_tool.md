# AI tool configurations
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
```