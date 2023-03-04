Observations:
- databricks auto scaling failure: 
Caused by: org.apache.spark.SparkException: Job aborted due to stage failure: Task 846 in stage 312.0 failed 4 times, most recent failure: Lost task 846.4 in stage 312.0 (TID 3896) (10.241.102.140 executor 13): ExecutorLostFailure (executor 13 exited caused by one of the running tasks) Reason: Command exited with code 50
- This error may happen after hours with several retries
- Cluster event: node lost

Analysis:
- Compare it with job run without autoscaling. The job completed successfully.
- Look up the error log, in which time the first time happens.
- Find out the first error was `WARN DLTDebugger: Failed to talk to RPC endpoint: dlt-debugger`
- Find out the reason for node lost was caused by spot instance we use

Solution:
- Use on-demand instance instead of spot instance
- The reason why fixed cluster doesn't have this problem is to be investigated.