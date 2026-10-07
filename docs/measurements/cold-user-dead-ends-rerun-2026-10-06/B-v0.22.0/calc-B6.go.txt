package main

import "fmt"

// add returns the sum of a and b.
func add(a, b int) int {
	return a + b
}

func main() {
	fmt.Println(add(2, 3))
}
