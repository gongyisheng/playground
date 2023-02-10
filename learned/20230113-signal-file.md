#### Why signal file is a bad idea
- signal file may be moved, renamed, or deleted
- signal file may be created by a different process
- signal file format may change  
#### On databricks, we use databricks api to trigger a new run. This is a better way than using signal file.