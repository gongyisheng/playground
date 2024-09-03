# ElasticSearch issues
## Add replica number
Modify replica number
```
PUT /your_index/_settings
{
  "index": {
    "number_of_replicas": 2  // Or whatever number works best for your cluster size
  }
}
```
In this case, you will have 3 nodes in total (2 replica + 1 master)