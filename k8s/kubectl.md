# kubectl wiki
## create deployment
`kubectl create deployment nginx-deployment --image=nginx`  
`kubectl expose deployment nginx-deployment --port=80 --type=NodePort`  
notes:  
1. command1 creates an deployment called nginx
2. command2 exposes port 80 to deployment, avaiable types:
  - **ClusterIP**: This is the default service type in Kubernetes. It exposes the service on a cluster-internal IP. The service is only accessible from within the cluster. It is suitable for internal services that need to communicate with each other within the cluster.

  - **NodePort**: This type exposes the service on a static port on each Node in the cluster. It creates a high-port (typically in the range 30000-32767) on each node, and this port is forwarded to the service. NodePort allows external clients to access the service using the Node's IP address and the allocated port.

   - **LoadBalancer**: This type provisions a load balancer for the service. It automatically assigns an external IP address to access the service and distributes incoming traffic among the pods of the service. LoadBalancer services are primarily used for services that need to be externally accessible and require load balancing.

   - **ExternalName**: This type maps the service to the contents of the `externalName` field (e.g., a CNAME record). ExternalName services are used for mapping services to external DNS names.

   - **Headless**: This type gives direct access to the pods backing the service without load balancing or a cluster IP. It is useful for stateful services that manage their own clustering and scaling.
## view service / deployment / pod
`kubectl get services`
`kubectl get deployments`
`kubectl get pods`