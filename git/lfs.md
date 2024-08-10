# git lfs
for files that are too large for git to handle, use git lfs.
## install 
ref: https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md
`curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`  
`sudo apt-get install git-lfs`  
## setup
`git lfs install`  
`git lfs track "*.psd"`  
`git add .gitattributes`  
## usage
`git add file.psd`  
`git commit -m "Add design file"`  
`git push origin main`  