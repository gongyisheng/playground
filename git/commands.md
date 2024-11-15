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