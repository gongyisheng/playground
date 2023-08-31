package learn_httptest

import (
	"net/http"
    "net/http/httptest"
    "testing"
	"fmt"
    "io"
    "log"
)

func TestDemo(t *testing.T) {
	// Create a new HTTP client.
    client := &http.Client{}

	// Create a new GET request.
    req, err := http.NewRequest(http.MethodGet, "https://www.google.com", nil)
    if err != nil {
        fmt.Println(err)
        return
    }

	// Send the request and get the response.
    resp, err := client.Do(req)
    if err != nil {
        fmt.Println(err)
        return
    }

	// Close the response body when we're done with it.
    defer resp.Body.Close()

    // Print the response status code.
    fmt.Println(resp.StatusCode)

    // response := httptest.NewRecorder()

    // got := response.Body.String()
    // want := "Hello, Foo!"
    // if got != want {
    //     t.Errorf("got %q, want %q", got, want)
    // }
}

func TestDemo2(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, client")
	}))
	defer ts.Close()

	res, err := http.Get(ts.URL)
	if err != nil {
		log.Fatal(err)
	}
	greeting, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s", greeting)
}