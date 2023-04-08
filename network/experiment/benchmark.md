Experiment benchmark settings
1. Start 2 EC2 instance on AWS in the same region, one as server, one as client.
2. Edit security group of server, add inbound rule: `Custom TCP Rule`, `Port Range: 8089`
3. Start server: `python -m http.server 8089`
4. Client can get files through `curl`: `curl http://<ip>:8089/<file_path>`