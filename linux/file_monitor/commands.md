## find file under current directory
`find ./ -iname "stdio.h"`
## find file using xcode-select (global)
`find $(xcode-select --print-path) -name stdio.h`
## find bin file
`whereis gcc`