/Users/temp/Documents/playground/llm/prompt_tool/backend_app/backend-config.example.yaml ---> /etc/prompt_tool/
# docker build
```
docker build -t chatbackend-test .
```

# docker run
# locally
```
docker run -d \
--name chatbackend-test-v3 \
-p 5600:5600 \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/backend-config.example.yaml:/etc/prompt_tool/backend-config.yaml" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/test-app-data.db:/var/data/app-data.db" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/test-key-data.db:/var/data/key-data.db" \
chatbackend-test:latest
```
# on raspberrypi
```
docker run -d \
--name chatbackend-test-v3 \
-p 5600:5600 \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/backend-config.example.yaml:/etc/prompt_tool/backend-config.yaml" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/test-app-data.db:/var/data/app-data.db" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/test-key-data.db:/var/data/key-data.db" \
chatbackend-test:latest
```