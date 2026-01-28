# Ray
```
# install
pip install -U "ray[data,train,tune,serve]"

# env
export RAY_memory_usage_threshold=0.9

# command
## on cpu machine
ray start --head --dashboard-host=0.0.0.0 --port=6378 --dashboard-port=8265 --metrics-export-port=9089 --ray-client-server-port=10001 --num-cpus 8

## on gpu machine
ray start --head --dashboard-host=0.0.0.0 --port=6378 --dashboard-port=8265 --metrics-export-port=9089 --ray-client-server-port=10001 --num-cpus 8 --num-gpus 1
```