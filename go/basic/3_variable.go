package main

import "fmt"

func main() {
    var a = "initial"
    fmt.Println(a)

    var b, c int = 1, 2
    fmt.Println(b, c)

    var d = true
    fmt.Println(d)

    var e int
    fmt.Println(e) // default 0

	var e_2 bool
	fmt.Println(e_2) // default false

    f := "apple"
    fmt.Println(f)

	a = "initialize again"
	fmt.Println(a)
}