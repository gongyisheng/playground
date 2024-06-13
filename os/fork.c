
int main(){
    int pid = fork();
    if(pid > 0){
        printf("parent: child=%d\n", pid);
        pid = wait();
        printf("child %d is done\n", pid);
    } else if(pid == 0){
        printf("child: exiting\n");
        exit();
    } else {
        printf("fork error\n");
    }
    return 0;
}