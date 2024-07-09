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
- store credentials
`git config --global credential.helper store`
- sign commit
`gpg --list-secret-keys --keyid-format=long`  
`git config --global user.signingkey <key>`  
ref: https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key  