- Ignore all  
`*`

- Unignore all with extensions  
`!*.*`

- Unignore all dirs  
`!*/`

#### Above combination will ignore all files without extension 

- Ignore files with extension `.class` & `.sm`  
`*.class`  
`*.sm`

- Ignore `bin` dir  
`bin/`  
 or  
`*/bin/*`

- Unignore all `.jar` in `bin` dir  
`!*/bin/*.jar`

- Ignore all `library.jar` in `bin` dir
`*/bin/library.jar`

- Ignore a file with extension  
`relative/path/to/dir/filename.extension`

- Ignore a file without extension  
`relative/path/to/dir/anotherfile`

#### After update gitignore, run below command to remove all files from git cache:
- remove cache and trace files again   
 `git rm -r --cached .`  
 `git add .`