//go:build linux

package gpudev

import (
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
)

// peakBandwidthGBs derives the card's theoretical memory bandwidth from the
// DEVICE, not from a table: memory clock (kHz) × bus width (bits) × 2 for DDR.
// On a 2070 SUPER that is 7001 MHz × 256 bit × 2 = 448 GB/s, matching spec.
//
// Deriving it matters. The trap this file exists to catch produced a reading of
// 8145 GB/s, and what made that obviously wrong was not intuition — it was having
// a ceiling to compare against. A hardcoded 448 would be a lie on any other card
// and would silently stop checking anything the moment the box changed.
func peakBandwidthGBs(t *testing.T, d *Device) float64 {
	t.Helper()
	clkKHz, err := d.dev.Attribute(gc.DeviceAttributeMemoryClockRate)
	if err != nil {
		t.Skipf("no memory clock attribute, cannot derive a ceiling: %v", err)
	}
	busBits, err := d.dev.Attribute(gc.DeviceAttributeGlobalMemoryBusWidth)
	if err != nil {
		t.Skipf("no bus width attribute, cannot derive a ceiling: %v", err)
	}
	if clkKHz <= 0 || busBits <= 0 {
		t.Skipf("implausible device attributes (clock=%d kHz, bus=%d bit)", clkKHz, busBits)
	}
	// kHz × bits → bytes/s: ×1e3 for kHz, /8 for bits→bytes, ×2 for DDR.
	return float64(clkKHz) * 1e3 * float64(busBits) / 8 * 2 / 1e9
}

// trafficGBs converts bytes COPIED into memory TRAFFIC, which is what the ceiling
// above actually bounds.
//
// WHY THIS FUNCTION EXISTS — IT IS A FACTOR OF TWO, AND IT BIT ME. A device-to-
// device copy reads n bytes and writes n bytes, so it moves 2n across the memory
// bus. peakBandwidthGBs is derived from clock × bus width, which is a bus-traffic
// figure. Comparing bytes-copied against it therefore understates utilisation by
// exactly 2×: the first version of this test measured a settled 166.6 GB/s on a
// 2070 SUPER and reported "37% of peak" for a copy that was really running at
// 333 GB/s of traffic, i.e. 74% — which is what the originating goinfer
// measurement meant when it said 347 GB/s / 78%.
//
// Both numbers are honest and neither is wrong; they answer different questions.
// Bytes-copied is what a caller budgets against ("how long to move my 20 MiB").
// Traffic is what the hardware ceiling bounds. Reporting only one of them, against
// the other's ceiling, is how two repos end up quoting numbers that differ by 2×
// forever without either being mistaken.
func trafficGBs(copiedGBs float64) float64 { return copiedGBs * 2 }

// TestCopyDevice_bandwidthUnderPhysicalCeiling is the gate for this whole change,
// and its job is to FAIL on an impossible number rather than print one.
//
// THE TRAP, in full, because every part of it pointed the wrong way. gocudrv's
// CopyToDevice/CopyToDeviceAt dispatch cuMemcpyDtoD_v2, whose name and whose own
// doc comment ("It blocks until the copy completes") both suggest the call
// returns when the bytes have landed. The CUDA driver defines DtoD as
// ASYNCHRONOUS with respect to the host, so it does not. Timing the call alone on
// 20 MiB measures ~9 µs of dispatch instead of ~116 µs of transfer, which
// computes to 8145 GB/s. That is not a suspicious throughput — it is roughly 18×
// a 2070 SUPER's physical maximum, i.e. impossible, and comparing against a
// device-derived ceiling is what surfaced it.
//
// So this test times copy AND synchronize together, and asserts the result is
// physically possible. A version that only printed the number would have printed
// 8145 GB/s and been believed.
func TestCopyDevice_bandwidthUnderPhysicalCeiling(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()

	peak := peakBandwidthGBs(t, d)
	t.Logf("device %s — derived peak %.1f GB/s", d.Name(), peak)

	const n = 20 << 20 // 20 MiB, the motivating snapshot size
	src := d.NewBufferBytes(n)
	dst := d.NewBufferBytes(n)
	if err := Upload(src, make([]byte, n)); err != nil {
		t.Fatalf("seed upload: %v", err)
	}

	// One untimed pass so the first copy's lazy context work is not in the sample.
	if err := CopyDevice(dst, src, n); err != nil {
		t.Fatalf("warmup copy: %v", err)
	}

	const reps = 20
	start := time.Now()
	for range reps {
		// CopyDevice synchronizes internally, which is exactly the point: timing
		// it is timing copy+sync. If that ever becomes async, this measurement
		// silently starts reporting dispatch again — hence the ceiling assert.
		if err := CopyDevice(dst, src, n); err != nil {
			t.Fatalf("copy: %v", err)
		}
	}
	el := time.Since(start)

	gbs := float64(n) * reps / el.Seconds() / 1e9
	traffic := trafficGBs(gbs)
	t.Logf("contiguous %d MiB × %d: %.3f ms/copy, %.1f GB/s copied = %.1f GB/s traffic (%.0f%% of %.1f peak)",
		n>>20, reps, float64(el.Microseconds())/1e3/reps, gbs, traffic, traffic/peak*100, peak)

	if traffic > peak {
		t.Fatalf("device-to-device moved %.1f GB/s of traffic (%.1f GB/s copied), above the device's physical "+
			"ceiling of %.1f GB/s — the copy is almost certainly not being awaited (cuMemcpyDtoD is async "+
			"wrt the host); this is the 8145 GB/s failure mode", traffic, gbs, peak)
	}
	if gbs <= 0 {
		t.Fatalf("measured %.1f GB/s — the timer or the copy did nothing", gbs)
	}
}

// TestCopyDeviceBatch_bandwidthUnderPhysicalCeiling covers the shape the consumer
// actually issues, which is NOT the contiguous one. The real caller snapshots 18
// layers × two buffers = 36 copies of 73 KB and 1 MiB. Reproduced on a 2070
// SUPER: 172.5 GB/s of traffic (39% of peak, median of 5) against 331 GB/s (74%)
// for one contiguous copy of the same bytes — so benchmarking only the big copy
// overstates what callers see by ~1.9×. Both shapes are gated and both reported.
//
// This shape is also the NOISY one: across five idle-box runs the contiguous
// number held to ±2% while this one ranged 133–207 GB/s. That is consistent with
// it being dispatch-bound rather than bandwidth-bound — it is measuring scheduler
// latency 36 times, not the memory system once. Take a median, not a best.
func TestCopyDeviceBatch_bandwidthUnderPhysicalCeiling(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()

	peak := peakBandwidthGBs(t, d)

	// 18 layers × (1 MiB + 73 KB), laid out with a gap between the two per-layer
	// buffers so coalesce CANNOT merge them — this measures the honest
	// many-dispatch shape rather than accidentally collapsing to one copy.
	const (
		layers = 18
		bigSz  = 1 << 20
		smlSz  = 73 << 10
		stride = bigSz + smlSz + 4096 // the gap
	)
	total := layers * stride
	src := d.NewBufferBytes(total)
	dst := d.NewBufferBytes(total)
	if err := Upload(src, make([]byte, total)); err != nil {
		t.Fatalf("seed upload: %v", err)
	}

	var copies []DeviceCopy
	bytes := 0
	for i := range layers {
		base := i * stride
		copies = append(copies,
			DeviceCopy{Dst: dst.At(base), Src: src.At(base), Bytes: bigSz},
			DeviceCopy{Dst: dst.At(base + bigSz + 2048), Src: src.At(base + bigSz + 2048), Bytes: smlSz},
		)
		bytes += bigSz + smlSz
	}
	if got := len(coalesce(copies)); got != len(copies) {
		t.Fatalf("this shape is meant to resist coalescing: %d pairs became %d copies", len(copies), got)
	}

	if err := CopyDeviceBatch(copies); err != nil {
		t.Fatalf("warmup batch: %v", err)
	}
	const reps = 20
	start := time.Now()
	for range reps {
		if err := CopyDeviceBatch(copies); err != nil {
			t.Fatalf("batch: %v", err)
		}
	}
	el := time.Since(start)

	gbs := float64(bytes) * reps / el.Seconds() / 1e9
	traffic := trafficGBs(gbs)
	t.Logf("batch %d copies (%d MiB) × %d: %.3f ms/batch, %.1f GB/s copied = %.1f GB/s traffic (%.0f%% of %.1f peak)",
		len(copies), bytes>>20, reps, float64(el.Microseconds())/1e3/reps, gbs, traffic, traffic/peak*100, peak)

	if traffic > peak {
		t.Fatalf("batched device-to-device moved %.1f GB/s of traffic (%.1f GB/s copied), above the physical "+
			"ceiling of %.1f GB/s — the batch is not being awaited", traffic, gbs, peak)
	}
}

// BenchmarkCopyDevice_contiguous and BenchmarkCopyDevice_manySmall are the two
// shapes the doc comments quote. They exist so the numbers in cuda_copy.go can be
// re-derived on another card rather than trusted from this one.
func BenchmarkCopyDevice_contiguous(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()

	const n = 20 << 20
	src := d.NewBufferBytes(n)
	dst := d.NewBufferBytes(n)
	b.SetBytes(n)
	b.ResetTimer()
	for b.Loop() {
		if err := CopyDevice(dst, src, n); err != nil {
			b.Fatalf("copy: %v", err)
		}
	}
}

func BenchmarkCopyDevice_manySmall(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()

	const (
		layers = 18
		bigSz  = 1 << 20
		smlSz  = 73 << 10
		stride = bigSz + smlSz + 4096
	)
	total := layers * stride
	src := d.NewBufferBytes(total)
	dst := d.NewBufferBytes(total)
	var copies []DeviceCopy
	bytes := 0
	for i := range layers {
		base := i * stride
		copies = append(copies,
			DeviceCopy{Dst: dst.At(base), Src: src.At(base), Bytes: bigSz},
			DeviceCopy{Dst: dst.At(base + bigSz + 2048), Src: src.At(base + bigSz + 2048), Bytes: smlSz},
		)
		bytes += bigSz + smlSz
	}
	b.SetBytes(int64(bytes))
	b.ResetTimer()
	for b.Loop() {
		if err := CopyDeviceBatch(copies); err != nil {
			b.Fatalf("batch: %v", err)
		}
	}
}
