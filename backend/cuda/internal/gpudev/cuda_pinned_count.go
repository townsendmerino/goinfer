//go:build linux

package gpudev

// PinnedHostBuffers is how many page-locked host buffers the device holds now: every NewHostBuffer and mapped host buffer not yet closed.
// A test reads it before and after an operation to prove the operation pins nothing.
func (d *Device) PinnedHostBuffers() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.hosts)
}
