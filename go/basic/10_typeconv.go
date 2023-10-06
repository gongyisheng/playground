package main

import (
	"fmt"
	"context"
)

type Entry struct {
	key string
	value string
	Context context.Context
}

func typeconv (e interface{}) {
	fmt.Println(e.(Entry).key)
	fmt.Println(e.(Entry).Context)
}

func main() {
	ctx := context.Background()
	ctx = context.WithValue(ctx, "cba", "abc")
	e := Entry{"key", "value", ctx}
	typeconv(e)
}