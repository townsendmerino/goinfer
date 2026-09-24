package decoder

import (
	"context"
	"testing"
)

// LFM2 with int8 KV (Options.KVQuant == "i8"): the int8 store must not be enabled for a family whose
// attention sizes its scores buffer from the f32 key store (forward_lfm2.go reads len(cache.Keys)),
// or the first decode step's attendQueryI8 writes past a zero-length scores buffer.
func TestLFM2_kvQuantI8_generates(t *testing.T) {
	const ckpt = "../testdata/lfm2-tiny"
	m, err := Load(ckpt, Options{Backend: "cpu", KVQuant: "i8"})
	if err != nil {
		t.Skipf("no lfm2 fixture: %v", err)
	}
	defer m.Close()
	stream, gen := m.Generate(context.Background(), []int{1, 2, 3, 4}, 6, SamplingParams{})
	n := 0
	for range stream {
		n++
	}
	if gen.Err() != nil {
		t.Fatalf("generate: %v", gen.Err())
	}
	if n == 0 {
		t.Fatal("no tokens generated")
	}
}
