package main

import (
	"fmt"
	"sync"
	"time"
)

func main() {
	var wg sync.WaitGroup

	// Set the number of goroutines to wait for
	wg.Add(2)

	// Goroutine 1
	go func() {
		defer wg.Done() // Decrement the counter when the goroutine completes
		time.Sleep(2 * time.Second)
		fmt.Println("Goroutine 1 completed")
	}()

	// Goroutine 2
	go func() {
		defer wg.Done() // Decrement the counter when the goroutine completes
		time.Sleep(1 * time.Second)
		fmt.Println("Goroutine 2 completed")
	}()

	// Wait for all goroutines to finish
	wg.Wait()
	fmt.Println("All goroutines completed")
}
