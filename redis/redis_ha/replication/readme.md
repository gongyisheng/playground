# Redis replication
## feature
- Replica instance will be exact copies of master instance
- Replica instance will be read-only, not accept writes
- Replica instance can accept connections from other replica
- Asynchronous replication is used by default. Replica also asynchronously ack the amount of data they receive.
- Synchronous replication of certain data can be requested by the clients using the WAIT command. 
- Replication is non-blocking on the master side, can accept writes when initializing replicas
- Replication is also largely non-blocking on the replica side. Replica can be configured to determine whether to serve reads of old data during initializing or return error to client.

## use cases
- scalability, slow O(N) commands can run on replica

## data safety
- STRONGLY RECOMMEND: turn on persistence in master and replica. If not, please avoid restart automatically. Auto restart + no persistence is dangerous, will cause all data lost.

## how it works
### metadata
- Replication ID: pseudo random string, that marks a given history of the dataset. Only change when restarts from scratch or promoted to master.
- Offset: A number that increments for every byte of replication stream that it is produced to be sent to replicas. Offset is incremented even no replica is connected
- Replication ID = big version, offset = small version.
- Replication ID is used to prevent that old master still lives after fail and new master already accepts writes. 

### replica init
- Use PSYNC command to send old replication ID and offset to determine whether partial resync or full resync is needed
- (Full resync process) master produces a RDB file and buffers all new writes, transfer RDB file to replica, store on disk and load to memory, send the rest of commands to replica

### failover
- When connected, master keeps sending a stream of commands to replica, including writes, key expires or evicted and other commands.
- When disconnected, replica try to proceed partial resynchronization, including commands that missed during disconnection.
- If partial resynchronization is not possible, replica asks for a full resynchronization.
- Replica automatically reconnect when connection is lost.
- When promote a replica to master, it will accept connections from other replica with a new replication ID, and prevent from doing full resync.
  
## configuration
```
replicaof <masterip> <masterport>

// auth
masteruser <username>
masterauth <master-password>

// replica settings
replica-serve-stale-data yes // if no, reply error when master is down.
replica-read-only yes        // read-only, can't accept write

// master settings
repl-diskless-sync yes      
repl-diskless-sync-delay 5
repl-ping-replica-period 10  // master send ping to replica
repl-timeout 60              // 
repl-disable-tcp-nodelay no  // disable TCP_NODELAY? 

// require at least 3 replicas with a lag <= 10 seconds use
min-replicas-to-write 3
min-replicas-max-lag 10
```