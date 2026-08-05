//go:build gpu

package gpu

// Deprecated Release() aliases (audit B-12). The GPU resource types now standardize on
// Close() error so they satisfy io.Closer and callers can write generic cleanup — the fix
// the audit asked for ("None implement io.Closer"). Release() is retained ONLY as a thin
// deprecated alias: some teardown paths keep a MIXED cleanup list of our types and raw
// wgpu objects (which expose Release(), not Close()), so a hard removal would force those
// lists to be rewritten. New code should call Close(); Release is removed in a future major.

// Deprecated: use Close.
func (r *DecodeRunner) Release() { _ = r.Close() }

// Deprecated: use Close.
func (d *DeviceBuffer) Release() { _ = d.Close() }

// Deprecated: use Close.
func (r *GEMVRunner) Release() { _ = r.Close() }

// Deprecated: use Close.
func (rm *ResidentW4A8) Release() { _ = rm.Close() }

// Deprecated: use Close.
func (rm *ResidentMatrix) Release() { _ = rm.Close() }

// Deprecated: use Close.
func (s *ResidentStackedW8A8) Release() { _ = s.Close() }

// Deprecated: use Close.
func (rm *ResidentW8A8) Release() { _ = rm.Close() }
