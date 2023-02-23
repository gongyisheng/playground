### Problem:
How to add new functionality to several existing objects without altering its structure?
### Solution:
You can choose inheritance or composition to add new functionality to existing objects. But inheritance is static and composition is dynamic. Decorator pattern is a good choice to add new functionality to existing objects dynamically.
### When to use:
When you want to add new functionality to existing objects dynamically.
### Steps:
Create an interface for the objects that will have additional functionality.
### Example:
```
usage_data = {}

# decorator to track function call count
def track_count(metric):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _metric = metric + "_count"
            usage_data[_metric] = usage_data.get(_metric, 0) + 1
            print(f"record count: _metric={_metric}, data={usage_data[_metric]}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
```