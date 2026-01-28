# issues

## ConnectError("tcp open error", os { code: 24, message: "Too many open files" })
```
env: docker container, 8X6000 Pro
task: miles, qwen3-30b-a3b dapo training
error: ConnectError("tcp open error", os { code: 24, message: "Too many open files" })
root case: 
- rollout worker is trying to open a new TCP connection
- OS inside the container has reached the maximum number of open file descriptors (ulimit -n)
fix:
- add following config in docker command: --ulimit nofile=1048576:1048576
- verify inside docker container: ulimit -n
```