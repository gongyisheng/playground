package main

import "fmt"

func main() {
	// initialize
	data := map[string]string{
		"foo": "bar",
		"foo1": "bar1",
		"foo2": "bar2",
		"foo3": "bar3",
		"foo4": "bar4",
	}
	fmt.Println(data)

	// add / update
	data["baz"] = "qux"
	fmt.Println(data)

	// delete
	delete(data, "foo")
	fmt.Println(data)

	// length
	fmt.Printf("length: %d\n", len(data))

	// get key & check if a key exists
	v1, ok1 := data["foo1"]
	fmt.Println(v1, ok1)
	v2, ok2 := data["xxx"]
	fmt.Println(v2, ok2)

	// get all keys
	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}

	// Iterate over a map
	for k, v := range data {
		fmt.Println(k, v)
	}

	// Clear a map
	for k := range data {
		delete(data, k)
	}
	fmt.Println(data)
}