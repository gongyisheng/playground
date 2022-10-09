package main

import "fmt"

func main() {
    // concat string
    fmt.Println("go" + "lang")
    // number calculation
    fmt.Println("1+1 =", 1+1)
    // integer-float calculation
    fmt.Println("7.0+3.0 =", 7.0+3.0)
    fmt.Println("7.0+3 =", 7.0+3)
    fmt.Println("7*3 =", 7*3)
    fmt.Println("7.0*3.0 =", 7.0*3.0)
    fmt.Println("7.0/3.0 =", 7.0/3.0)
    fmt.Println("7/3 =", 7/3)
    fmt.Println("7.0/3 =", 7.0/3)
    fmt.Println("7/3.0 =", 7/3.0)
    // logic calculation
    fmt.Println(true && false)
    fmt.Println(true || false)
    fmt.Println(!true)
}