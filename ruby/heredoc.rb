#!/usr/bin/ruby -w
# -*- coding : utf-8 -*-
 
print <<EOF
This is the frist way to create here document 。
Multiple line string.
EOF
 
print <<"EOF";
This is the second way to create here document 。
Multiple line string.
EOF
 
print <<`EOC` # execute commands
echo hi there
echo lo there
EOC
 
print <<"foo", <<"bar"  # you can stack them
I said foo.
foo
I said bar.
bar