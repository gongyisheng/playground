use grep to search for a string in a file `-e`  and count the number of matches `-c`   
`grep -e "status\[200\]" ***.log -c`

use `zgrep` to search in a gzipped file   
`zgrep email_connections ***.log | grep "status\[400\]"`

search for all the files in a directory and subdirectories: `*`  
`zgrep email_connections * | grep "status\[400\]"`