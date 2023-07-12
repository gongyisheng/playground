package main

import (
	"fmt"
	"sync"
	"time"
)

func test(i int16, wg *sync.WaitGroup) {
	defer wg.Done() // Decrement the counter when the goroutine completes
	time.Sleep(2 * time.Second)
	fmt.Printf("Goroutine %d completed\n", i)
}

func main() {
	var wg sync.WaitGroup

	// Set the number of goroutines to wait for
	wg.Add(3)

	// Goroutine 1
	go test(1, &wg)
	// Goroutine 2
	go test(2, &wg)
	// Goroutine 3
	go test(3, &wg)
	// Wait for all goroutines to finish
	wg.Wait()

	fmt.Println("--------------------")

	// Set the number of goroutines to wait for
	wg.Add(3)
	// Goroutine 4
	go test(4, &wg)
	// Goroutine 5
	go test(5, &wg)
	// Goroutine 6
	go test(6, &wg)
	// Wait for all goroutines to finish
	wg.Wait()

	fmt.Println("All goroutines completed")
}
