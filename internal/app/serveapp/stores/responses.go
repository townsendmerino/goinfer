package stores

import (
	"slices"
	"sync"
)

// MaxResponseStoreBytes bounds a ResponseStore's estimated payload alongside its entry count (64 MiB).
const MaxResponseStoreBytes = 64 << 20

// ResponseEntry holds one stored /v1/responses conversation history for previous_response_id continuation.
type ResponseEntry[M any] struct {
	Model    string
	Messages []M
	size     int
}

// ResponseStore is a FIFO count- and byte-bounded store for /v1/responses conversation state.
type ResponseStore[M any] struct {
	mu       sync.Mutex
	m        map[string]*ResponseEntry[M]
	order    []string
	cap      int
	maxBytes int
	bytes    int
	sizeOf   func(model string, messages []M) int
}

// NewResponseStore constructs a ResponseStore holding at most cap entries and MaxResponseStoreBytes estimated payload
// bytes, measured by sizeOf.
func NewResponseStore[M any](cap int, sizeOf func(model string, messages []M) int) *ResponseStore[M] {
	return NewResponseStoreWithMaxBytes(cap, MaxResponseStoreBytes, sizeOf)
}

// NewResponseStoreWithMaxBytes constructs a ResponseStore with an explicit byte budget.
func NewResponseStoreWithMaxBytes[M any](cap, maxBytes int, sizeOf func(model string, messages []M) int) *ResponseStore[M] {
	return &ResponseStore[M]{
		m:        map[string]*ResponseEntry[M]{},
		cap:      cap,
		maxBytes: maxBytes,
		sizeOf:   sizeOf,
	}
}

// MaxBytes returns the store's byte budget.
func (s *ResponseStore[M]) MaxBytes() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxBytes
}

// Bytes returns the total estimated payload bytes currently held in the store.
func (s *ResponseStore[M]) Bytes() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bytes
}

// Get returns id's stored ResponseEntry, or nil if unknown or evicted.
func (s *ResponseStore[M]) Get(id string) *ResponseEntry[M] {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.m[id]
}

// Put stores (model, messages) under id as the newest entry, evicting the oldest entries until both count and byte
// bounds hold. An entry larger than maxBytes is refused without evicting other entries, while dropping any older entry
// stored under the same id so a continuation of a refused response never sees stale history.
func (s *ResponseStore[M]) Put(id, model string, messages []M) {
	sz := 0
	if s.sizeOf != nil {
		sz = s.sizeOf(model, messages)
	}
	e := &ResponseEntry[M]{Model: model, Messages: messages, size: sz}
	s.mu.Lock()
	defer s.mu.Unlock()
	if old, ok := s.m[id]; ok {
		s.bytes -= old.size
		delete(s.m, id)
		if i := slices.Index(s.order, id); i >= 0 {
			s.order = slices.Delete(s.order, i, i+1)
		}
	}
	if e.size > s.maxBytes {
		return
	}
	s.order = append(s.order, id)
	s.m[id] = e
	s.bytes += e.size
	for len(s.order) > s.cap || s.bytes > s.maxBytes {
		s.bytes -= s.m[s.order[0]].size
		delete(s.m, s.order[0])
		s.order = s.order[1:]
	}
}
