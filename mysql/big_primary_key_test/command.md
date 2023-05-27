## Stop saving the buffer pool to disk at shutdown
`SET GLOBAL innodb_buffer_pool_dump_at_shutdown=OFF;`
## Stop loading the buffer pool from disk at startup
`SET GLOBAL innodb_buffer_pool_load_at_startup=OFF;`

## Check the status of the dump and load
`SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_dump_status';`  
`SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_load_status';`

Reloading the buffer pool is very fast, and is performed in the background so the users will not be effected