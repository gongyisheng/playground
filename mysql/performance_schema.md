### time wait by event
```
SELECT
  EVENT_NAME,
  COUNT_STAR,
  SUM_TIMER_WAIT / 1000000000 as SUM_TIMER_WAIT_MS,
  AVG_TIMER_WAIT / 1000000000 as AVG_TIMER_WAIT_MS,
  MAX_TIMER_WAIT / 1000000000 as MAX_TIMER_WAIT_MS
FROM
  performance_schema.events_waits_summary_global_by_event_name
ORDER BY
  SUM_TIMER_WAIT_MS DESC
LIMIT
  50
```

### time wait by sql 
```
SELECT
  COUNT_STAR,
  SUM_TIMER_WAIT/1000000000 as SUM_TIMER_WAIT_MS,
  AVG_TIMER_WAIT/1000000000 as AVG_TIMER_WAIT_MS,
  MAX_TIMER_WAIT/1000000000 as MAX_TIMER_WAIT_MS,
  SUM_LOCK_TIME/1000000000 as SUM_LOCK_TIME_MS,
  DIGEST_TEXT,
  FIRST_SEEN,
  LAST_SEEN,
  SUM_ROWS_EXAMINED,
  SUM_CREATED_TMP_TABLES,
  SUM_CREATED_TMP_DISK_TABLES
  SUM_ROWS_SENT
FROM
  performance_schema.events_statements_summary_by_digest
ORDER BY
  SUM_TIMER_WAIT_MS DESC 
LIMIT 
  100;
```

## disk i/o by file
```
SELECT
  FILE_NAME,
  EVENT_NAME,
  COUNT_STAR,
  SUM_TIMER_WAIT/1000000000 as SUM_TIMER_WAIT_MS,
  AVG_TIMER_WAIT/1000000000 as AVG_TIMER_WAIT_MS,
  MAX_TIMER_WAIT/1000000000 as MAX_TIMER_WAIT_MS,
  SUM_TIMER_READ/1000000000 as SUM_TIMER_READ_MS,
  AVG_TIMER_READ/1000000000 as AVG_TIMER_READ_MS,
  MAX_TIMER_READ/1000000000 as MAX_TIMER_READ_MS,
  SUM_TIMER_WRITE/1000000000 as SUM_TIMER_WRITE_MS,
  AVG_TIMER_WRITE/1000000000 as AVG_TIMER_WRITE_MS,
  MAX_TIMER_WRITE/1000000000 as MAX_TIMER_WRITE_MS,
  COUNT_READ,
  COUNT_READ/COUNT_STAR as READ_PERC,
  ROUND(SUM_NUMBER_OF_BYTES_READ/1024/1024,2) as SUM_READ_MB,
  ROUND(SUM_NUMBER_OF_BYTES_READ/COUNT_READ/1024/1024,2) as AVG_READ_MB,
  COUNT_WRITE,
  COUNT_WRITE/COUNT_STAR as WRITE_PERC,
  ROUND(SUM_NUMBER_OF_BYTES_WRITE/1024/1024,2) as SUM_WRITE_MB,
  ROUND(SUM_NUMBER_OF_BYTES_WRITE/COUNT_WRITE/1024/1024,2) as AVG_WRITE_MB
FROM
  performance_schema.file_summary_by_instance
ORDER BY
  SUM_NUMBER_OF_BYTES_READ + SUM_NUMBER_OF_BYTES_WRITE DESC
```

# memory usage by event
```
SELECT
  EVENT_NAME,
  CURRENT_COUNT_USED,
  HIGH_COUNT_USED,
  CURRENT_NUMBER_OF_BYTES_USED / 1024 / 1024 as CURRENT_USED_MB,
  HIGH_NUMBER_OF_BYTES_USED / 1024 / 1024 as HIGH_USED_MB
FROM
  performance_schema.memory_summary_global_by_event_name
ORDER BY
  CURRENT_USED_MB DESC
```

```
SELECT
  event_name, 
  current_alloc, 
  high_alloc 
FROM 
  sys.memory_global_by_current_bytes 
WHERE 
  current_count > 0;
```

# high level memory usage
```
SELECT
  substring_index(substring_index(event_name, '/', 2), '/', -1) as event_type,
  round(sum(CURRENT_NUMBER_OF_BYTES_USED) / 1024 / 1024, 2) as MB_CURRENTLY_USED
FROM
  performance_schema.memory_summary_global_by_event_name
GROUP BY
  event_type
HAVING
  MB_CURRENTLY_USED > 0;
```

# buffer pool related
```
# config
show variables like 'innodb_buffer_pool%';
# status
show status like 'innodb_buffer_pool_read%';
# hit rate
innodb_buffer_pool_read_requests / (innodb_buffer_pool_read_requests + innodb_buffer_pool_reads) * 100 as buffer_pool_hit_rate
```