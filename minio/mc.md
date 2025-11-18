# minio client related
## install
```
wget https://dl.min.io/client/mc/release/linux-amd64/mc # x86
wget https://dl.min.io/client/mc/release/linux-arm64/mc # arm

chmod +x mc
sudo mv mc /usr/local/bin/
mc --version
```

## setup
```
mc alias set <ALIAS> <YOUR-S3-ENDPOINT> <YOUR-ACCESS-KEY> <YOUR-SECRET-KEY>
```

## commands
```
# copy
mc cp <SOURCE> <TARGET>
mc cp --recursive <SOURCE> <TARGET>

# move, note that the leaf folder will be created automatically
mc mv <SOURCE> <TARGET>
mc mv --recursive <SOURCE> <TARGET>
```