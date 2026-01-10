# Distributed Ops

## Quick Reference Diagram
```
Broadcast         Reduce          All-Reduce
[A]──┬──▶[A]     [A]──┐           [A]──┐
     ├──▶[A]     [B]──┼──▶[Σ]     [B]──┼──▶[Σ] (all ranks)
     ├──▶[A]     [C]──┤           [C]──┤
     └──▶[A]     [D]──┘           [D]──┘

Scatter           Gather          All-Gather
[ABCD]─┬─▶[A]    [A]──┐           [A]──┐
       ├─▶[B]    [B]──┼──▶[ABCD]  [B]──┼──▶[ABCD] (all ranks)
       ├─▶[C]    [C]──┤           [C]──┤
       └─▶[D]    [D]──┘           [D]──┘

Reduce-Scatter
[A0,A1,A2,A3]──┐
[B0,B1,B2,B3]──┼──▶ Rank0:[Σ0], Rank1:[Σ1], Rank2:[Σ2], Rank3:[Σ3]
[C0,C1,C2,C3]──┤
[D0,D1,D2,D3]──┘
```

## Broadcast
One → All (same data)
```
[A] → [A][A][A][A]
```

## Reduce
All → One (sum data)
```
[ABCD] → [A+B+C+D][][][]
```

## Scatter
One → All (split data)  
```
[ABCD] → [A][B][C][D] 
```

## Gather
All → One (collect data)
```
[A][B][C][D] → [ABCD][][][]
```

## All-Reduce 
All → All (reduce + broadcast)
```
[A][B][C][D] → [A+B+C+D][A+B+C+D][A+B+C+D][A+B+C+D]
```

## All-Gather
All → All  (gather + broadcast)
```
[A][B][C][D] → [ABCD][ABCD][ABCD][ABCD]
```

## Reduce-Scatter
All → All (reduce + scatter)
```
[A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3][D0 D1 D2 D3] → [A0+B0+C0+D0][A1+B1+C1+D1][A2+B2+C2+D2][A3+B3+C3+D3]
```

