import redis

# pipeline
r = redis.Redis(host='localhost', port=6379, db=0)
r.set('foo', 'bar', ex=600)
pipe = r.pipeline()
pipe.get('foo')
pipe.ttl('foo')
res = pipe.execute()
print(res[0])
print(res[1])