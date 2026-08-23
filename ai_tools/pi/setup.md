# Pi Agent

## install
```
curl -fsSL https://pi.dev/install.sh | sh
pi install npm:pi-permission-layers
```

## config
1. skip permissions
```
# update ~/.bashrc
export PI_PERMISSION_LEVEL=bypassed
```

2. set base url if use litellm
```
# set if you use litellm
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://<litellm-host>"
    }
  }
}
```

3. agents.md
```
# global
~/.pi/agent/AGENTS.md

# local
<dir>/AGENTS.md
<dir>/CLAUDE.md
<dir>/AGENTS.override.md
```

4. install skills
```
# skill folder path 
~/.pi/agent/skills/                                                                                             
~/.agents/skills/

# add claude/codex skills via settings.json
{
    "skills": [
        "~/.claude/skills",
        "~/.codex/skills",
    ]
}
# recommend to install anthropic-skills(https://github.com/anthropics/skills) 
```