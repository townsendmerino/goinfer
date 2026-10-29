//go:build cuda

package cuda

import "github.com/townsendmerino/goinfer/internal/mem"

// A test's own buffers inside a plain r.do job. Each helper leases on the test line for its one allocation only, so whatever the
// code under test allocates in the same job keeps the job's line, and the suite's I1 check still reports it when that line is
// none. Executor thread (inside r.do) or before any job runs.

func (r *cudaResident) tf(n int) Buffer          { defer r.onLine(mem.LineTest)(); return r.af(n) }
func (r *cudaResident) ti(n int) Buffer          { defer r.onLine(mem.LineTest)(); return r.ai(n) }
func (r *cudaResident) tup32(v []float32) Buffer { defer r.onLine(mem.LineTest)(); return r.up32(v) }
func (r *cudaResident) tupu32(v []uint32) Buffer { defer r.onLine(mem.LineTest)(); return r.upu32(v) }
func (r *cudaResident) tupu16(v []uint16) Buffer { defer r.onLine(mem.LineTest)(); return r.upu16(v) }
