```
deltaTable = DeltaTable.forPath(spark, "/path/to/table")
deltaTable = DeltaTable.convertToDelta(spark, "parquet.`/path/to/table`")
```