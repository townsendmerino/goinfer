// mem1-libload loads a model as a library caller does, decoder.Load with only a Quant set, and prints the greedy tokens
// for decoder's TestLoad_productionDefaultTakesTheSidecar: a test binary loads directly, so the default needs a program.
//
//	go run ./testdata/mem1-libload <model> <quant>
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
)

func main() {
	m, err := decoder.Load(os.Args[1], decoder.Options{Quant: os.Args[2]})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer m.Close()
	out, gen := m.Generate(context.Background(), []int{3, 14, 15, 9, 26, 5, 35}, 8, decoder.SamplingParams{})
	var ids []int
	for id := range out {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("tokens", fmt.Sprint(ids))
}
