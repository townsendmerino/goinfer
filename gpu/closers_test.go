//go:build gpu

package gpu

import (
	"io"
	"reflect"
)

// closers collects the wrappers a test uploads so one deferred closeAll releases them, newest first. Resident weights and device buffers are caller-owned (Context.Close does not free them), and the
// package's LEAK REPORT (leak_report_test.go) names every test that closed a Context with some still live, with the line that created each: a test that builds a model by hand adds what it uploads here.
type closers []io.Closer

// add records x (a nil wrapper is ignored: a failed Upload returns one) and returns it, so an upload site stays one expression.
func add[T io.Closer](c *closers, x T) T {
	if v := reflect.ValueOf(x); !v.IsValid() || (v.Kind() == reflect.Ptr && v.IsNil()) {
		return x
	}
	*c = append(*c, x)
	return x
}

func (c *closers) closeAll() { // pointer receiver: `defer own.closeAll()` must see what is added after the defer line
	for i := len(*c) - 1; i >= 0; i-- {
		_ = (*c)[i].Close()
	}
}
