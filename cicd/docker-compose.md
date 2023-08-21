# docker-compose
## docker-compose.yml
```yml
version: '3.8'

services:
  minio:
    image: wktk/minio-server
    ports:
      - 9000:9000
    environment:
      MINIO_ACCESS_KEY: "minioadmin"
      MINIO_SECRET_KEY: "minioadmin"
```
## run yml file in github actions
```bash
steps:
    - uses: actions/checkout@v3
    - name: Setup MinIO Server
    shell: bash
    working-directory: core/src/services/s3/fixtures
    run: docker-compose -f docker-compose-minio.yml up -d
```
## run yml file in command line
```
docker-compose -f docker-compose-minio.yml up -d
```
- `-f` means specifying yml file
- `up` means start service
- `-d` means run in background