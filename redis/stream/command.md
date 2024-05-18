# Redis Stream Commands
## XADD
```bash
XADD mystream * field1 value1 field2 value2
```
## XLEN
```bash
XLEN mystream
```
## XRANGE
```bash
XRANGE mystream - + COUNT 2
```
## XREAD
```bash
XREAD COUNT 2 STREAMS mystream 0
```
## XTRIM
```bash
XTRIM mystream MAXLEN 1000
```
## XDEL
```bash
XDEL mystream 12345
```
## XGROUP
```bash
XGROUP CREATE mystream mygroup $
```
## XINFO
```bash
XINFO STREAM mystream
```
## XREADGROUP
```bash
XREADGROUP GROUP mygroup consumer1 COUNT 1 STREAMS mystream >
```
## XPENDING
```bash
XPENDING mystream mygroup
```
## XACK
```bash
XACK mystream mygroup 12345
```
## XCLAIM
```bash
XCLAIM mystream mygroup consumer1 0 12345
```