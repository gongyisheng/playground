# minikube wiki
## start
`minikube start`
`minikube start &`: start in background
`minikube start --driver=podman`: start with podman driver

## minikube dashboard
`minikube dashboard`: open dashboard
`kubectl proxy --address='0.0.0.0' --accept-hosts='^*$' &`: start kubectl proxy in background