use grep to search for a string in a file `-e`  and count the number of matches `-c`   
`grep -e "status\[200\]" ***.log -c`

use `zgrep` to search in a gzipped file   
`zgrep email_connections ***.log | grep "status\[400\]"`

search for all the files in a directory and subdirectories: `*`  
`zgrep email_connections * | grep "status\[400\]"`

当前目录有多个文件夹，查找每个文件夹一个pattern的次数，比如每小时某个日志的数量
ls | xargs -I {} echo "find {} -name 1*.gz | xargs zgrep gmail | wc -l" | sh
用sed把部分字段拿出来
head 10-27_GAP.log | sed -e "s/^.*'account': '\([^']*\)'.*'password': '\([^']\+\).*$/\1 \2/"
查询一天里部分小时的日志
echo {14..23} | tr ' ' '\n' |  xargs -I {} echo "find {} -name *.gz"