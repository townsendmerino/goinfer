package main

import (
	"sync"
	"testing"
	"time"
)

// TestTryLockUntil is the C-22 gate: the shutdown checkpoint must never block forever on a busy model.
// tryLockUntil takes a free lock immediately, gives up at the deadline on a held lock (rather than
// hanging), and takes it once it's released before the deadline.
func TestTryLockUntil(t *testing.T) {
	var mu sync.Mutex

	// free → acquired immediately.
	if !tryLockUntil(&mu, time.Now().Add(time.Second)) {
		t.Fatal("tryLockUntil failed on a free mutex")
	}
	mu.Unlock()

	// held past the deadline → returns false without hanging.
	mu.Lock()
	start := time.Now()
	if tryLockUntil(&mu, time.Now().Add(80*time.Millisecond)) {
		t.Fatal("tryLockUntil acquired a mutex that was held the whole time")
	}
	if el := time.Since(start); el > time.Second {
		t.Fatalf("tryLockUntil blocked %v — did not honor the deadline", el)
	}
	mu.Unlock()

	// released before the deadline → acquired.
	mu.Lock()
	go func() { time.Sleep(30 * time.Millisecond); mu.Unlock() }()
	if !tryLockUntil(&mu, time.Now().Add(2*time.Second)) {
		t.Fatal("tryLockUntil gave up on a mutex freed before the deadline")
	}
	mu.Unlock()
}
