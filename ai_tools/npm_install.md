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
