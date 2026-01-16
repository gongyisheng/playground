# Bash Line Learned

## Continuation and Comments

Don't put comments inside a line-continuation chain:

```bash
# BAD - comment breaks the chain
cmd arg1 \
# arg2 \
arg3      # arg3 is now part of the comment!

# GOOD - remove the line entirely
cmd arg1 \
arg3
```

**Why**: Backslash `\` joins lines, so `arg1 \ # arg2 \ arg3` becomes `arg1 # arg2 arg3` - everything after `#` is ignored.
