- pull commits after rebase  
`git pull --rebase`
- revert commit  
`git reset --hard HEAD~`
- revert commit and keep changes  
`git reset --soft HEAD~`
- revert commit from github  
`git push origin HEAD --force`
- force push  
`git push --force`
- edit commit message  
`git commit --amend`
- look up commit by hash
`git checkout <branch>`
`git log -p -1 <hash>`
- check commit diff
`git show <hash>`
- revert commit
`git revert <hash>`
`git reset --soft <hash>`
- undo unsaved changes
`git stash`
- set username and email
`git config --global user.name XXX`
`git config --global user.email XXX`
- store credentials
`git config --global credential.helper store`
- reset credentials
`git config --global --unset credential.helper`
- sign commit
`gpg --list-secret-keys --keyid-format=long`  
`git config --global user.signingkey <key>`  
`git config commit.gpgsign false`  
ref: https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key  
- upload big files
`git config --global http.postBuffer 524288000`  
original error:
```
error: RPC failed; HTTP 408 curl 22 The requested URL returned error: 408
send-pack: unexpected disconnect while reading sideband packet
Writing objects: 100% (5/5), 1.76 GiB | 5.67 MiB/s, done.
```
- add submodule
`git submodule add <repo-url> <path/to/submodule>`
- pull submodule 
`git submodule update --init --recursive`
- pull along with submodule
`git pull --recurse-submodules`
- set config to always pull submodule
`git config submodule.recurse true`
- clone other people's fork branch to my fork
```
git remote add <their-name> https://github.com/<their-username>/<repo>.git
git fetch <their-name>
git checkout -b <branch-name> <their-name>/<branch-name>
git push origin <branch-name>
```

setup commands
```
git config --global user.name gongyisheng
git config --global user.email yishenggong9437@gmail.com
git config --global credential.helper store
```