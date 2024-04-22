# docker build
```
docker build -t chatbackend .
docker build -t chatbackend-test .
```

# docker run
# local test
```
docker run -d \
--name chatbackend-test \
-p 5600:5600 \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/backend-config.example.yaml:/etc/prompt_tool/backend-config.yaml" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/test-app-data.db:/var/data/app-data.db" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/test-key-data.db:/var/data/key-data.db" \
-v "/Users/temp/Documents/playground/llm/prompt_tool/backend_app/backend-app.log:/var/log/prompt_tool/backend-app.log" \
chatbackend-test:latest
```
# on raspberrypi
```
docker run -d \
--name chatbackend \
-p 5600:5600 \
-v "~/user-key/prompt_tool/backend-app.prod.yaml:/etc/prompt_tool/backend-config.yaml" \
-v "/var/data/app-data.db:/var/data/app-data.db" \
-v "/var/data/key-data.db:/var/data/key-data.db" \
-v "~/log/chatbackend.log:/var/log/prompt_tool/backend-app.log" \
chatbackend:latest
```