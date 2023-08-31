package learn_httptest

import (
    "fmt"
    "net/http"
)

func helloServer(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Path[1:]
    fmt.Fprintf(w, "Hello, %s!", name)
}

func main() {
    http.HandleFunc("/", helloServer)
    http.ListenAndServe(":8080", nil)
}