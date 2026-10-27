package main

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

func main() {
	old, _ := os.ReadFile(os.Args[1])
	assembled, err := paritykey.ManifestJSON(paritykey.Dir("."))
	if err != nil {
		panic(err)
	}
	var a, b any
	json.Unmarshal(old, &a)
	json.Unmarshal(assembled, &b)
	fa := a.(map[string]any)["families"].(map[string]any)
	fb := b.(map[string]any)["families"].(map[string]any)
	diff := 0
	for k, v := range fa {
		if !reflect.DeepEqual(v, fb[k]) {
			fmt.Println("DIFF", k)
			diff++
		}
	}
	for k := range fb {
		if _, ok := fa[k]; !ok {
			fmt.Println("EXTRA", k)
			diff++
		}
	}
	setsEqual := reflect.DeepEqual(a.(map[string]any)["shared_sets"], b.(map[string]any)["shared_sets"])
	fmt.Printf("families old %d records %d; differing %d; shared sets equal %v\n", len(fa), len(fb), diff, setsEqual)
	if diff == 0 && setsEqual {
		fmt.Println("R1 PASS")
	} else {
		fmt.Println("R1 FAIL")
	}
}
