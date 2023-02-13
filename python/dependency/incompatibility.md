ERROR INFO:  
```
OSError: Cannot load native module 'Crypto.Hash._SHA256': Cannot load '_SHA256.so': cannot load library '/Users/temp/.pyenv/versions/2.7.18/lib/python2.7/site-packages/Crypto/Util/../Hash/_SHA256.so': dlopen(/Users/temp/.pyenv/versions/2.7.18/lib/python2.7/site-packages/Crypto/Util/../Hash/_SHA256.so, 0x0002): tried: '/Users/temp/.pyenv/versions/2.7.18/lib/python2.7/site-packages/Crypto/Util/../Hash/_SHA256.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e')), '/Users/temp/.pyenv/versions/2.7.18/lib/python2.7/site-packages/Crypto/Hash/_SHA256.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e')).  Additionally, ctypes.util.find_library() did not manage to locate a library called '/Users/temp/.pyenv/versions/2.7.18/lib/python2.7/site-packages/Crypto/Util/../Hash/_SHA256.so', Not found '_SHA256module.so'
```
SOLUTION:
```
ARCHFLAGS="-arch arm64" pip install pycrypto --compile --no-cache-dir
```