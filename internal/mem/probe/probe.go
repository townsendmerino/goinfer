package probe

import "sync"

// Reader is the machine readings a decision takes. Host reads this machine; a Fake returns what a test sets. Every
// method follows the package's rule: 0, or ok false, means unknown, and the caller proceeds.
type Reader interface {
	Physical() int64
	Available() int64
	SwapUsed() (usedBytes int64, ok bool)
	SelfSwap() (usedBytes int64, ok bool)
	FaultCounts() (minflt, majflt int64, ok bool)
	ResidentFraction(mapping []byte) float64
}

// Host is the Reader over this machine: each method is the package function of the same name. SelfSwap is SelfSwapReader's
// reader, or unknown where the platform has none.
var Host Reader = host{}

type host struct{}

func (host) Physical() int64                         { return Physical() }
func (host) Available() int64                        { return Available() }
func (host) SwapUsed() (int64, bool)                 { return SwapUsed() }
func (host) FaultCounts() (int64, int64, bool)       { return FaultCounts() }
func (host) ResidentFraction(mapping []byte) float64 { return ResidentFraction(mapping) }

func (host) SelfSwap() (int64, bool) {
	r, _ := SelfSwapReader()
	if r == nil {
		return 0, false
	}
	return r()
}

// Fake is a Reader whose readings a test sets. Its zero value reads unknown everywhere except ResidentFraction, which
// reads 0 (cold); Set changes a reading while other goroutines read it. A swap or fault reading is known when its OK
// field is true.
type Fake struct {
	mu                   sync.Mutex
	PhysicalBytes        int64
	AvailableBytes       int64
	SwapUsedBytes        int64
	SwapUsedOK           bool
	SelfSwapBytes        int64
	SelfSwapOK           bool
	MinorFaults          int64
	MajorFaults          int64
	FaultsOK             bool
	ResidentFractionRead float64
}

// Set runs f with the fake locked, so a test can change readings while a watcher or a decision reads them.
func (f *Fake) Set(fn func(f *Fake)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fn(f)
}

func (f *Fake) Physical() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.PhysicalBytes
}

func (f *Fake) Available() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.AvailableBytes
}

func (f *Fake) SwapUsed() (int64, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.SwapUsedBytes, f.SwapUsedOK
}

func (f *Fake) SelfSwap() (int64, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.SelfSwapBytes, f.SelfSwapOK
}

func (f *Fake) FaultCounts() (int64, int64, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.MinorFaults, f.MajorFaults, f.FaultsOK
}

func (f *Fake) ResidentFraction([]byte) float64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.ResidentFractionRead
}
