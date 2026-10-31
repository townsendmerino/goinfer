//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// M1′, the model of Vulkan's device memory that holds on nobara (docs/measurements/sb2-vk-2026-10-31/README.md, "M1′'s held-out
// pass"): gpu-allocator 0.28.0's placement under wgpu-native's Performance hints, one 64 MiB host-visible block, and 144 MiB for
// the device itself. score_vk1.py's m1p_predict is the same model; TestVkM1p_matchesTheScorer holds the two to the byte.
const (
	vkMiB         = int64(1) << 20
	vkInternal    = 512<<10 + 256 + 256 // wgpu-core's own per-device buffers: the zero buffer, external-texture params, indirect validation
	vkAlign       = 256                 // the driver's buffer alignment, assumed (M1's source reading)
	vkDevFirst    = 128 * vkMiB
	vkDevMax      = 256 * vkMiB
	vkHostFirst   = 64 * vkMiB
	vkHostMax     = 128 * vkMiB
	vkHostKept    = 64 * vkMiB  // gpu-allocator keeps a memory type's last block: the upload staging's
	vkDeviceConst = 144 * vkMiB // fitted on G-VK1's six loads, judged on five held-out
)

// vkCUDAContext is the CUDA context a profile's cuda_free_bytes was read inside: the probe's figure excludes it and a Vulkan load does
// not pay it. Measured on nobara's card (G-VK1); a discrete profile not listed keeps the census's previous line.
var vkCUDAContext = map[string]int64{"nobara-desktop": 105 * vkMiB, "nobara-no-desktop": 105 * vkMiB}

// vkCapacityPlantForTest is subtracted from every restated discrete capacity, and vkCapacityTightForTest sets each capacity one byte
// short of slot 0's blocks and the probe: G-VR2's plants.
var (
	vkCapacityPlantForTest int64
	vkCapacityTightForTest bool
)

// vkAligned is a buffer's size as gpu-allocator is asked for it: wgpu-core's 4-byte padding, then the driver's alignment.
func vkAligned(n int64) int64 {
	n = (max(n, 4) + 3) / 4 * 4
	return (n + vkAlign - 1) / vkAlign * vkAlign
}

// vkPacker is gpu-allocator's placement on one memory type: a request larger than the current block size gets a block of its own,
// of exactly its size; any other goes to the first open block it fits, newest first, or opens a block. total is the bytes allocated.
type vkPacker struct {
	first, max int64
	free       []int64
	total      int64
}

func (p *vkPacker) place(r int64) {
	mb := min(p.first<<min(len(p.free), 7), p.max)
	if r > mb {
		p.total += r
		return
	}
	for i := len(p.free) - 1; i >= 0; i-- {
		if p.free[i] >= r {
			p.free[i] -= r
			return
		}
	}
	p.free = append(p.free, mb-r)
	p.total += mb
}

func (p vkPacker) clone() vkPacker {
	p.free = append([]int64(nil), p.free...)
	return p
}

// vkM1p is M1′'s device use for buffers placed in order, on a device with a host-visible device-local memory type (nobara's BAR
// heap). mapRead buffers live in host memory and are not counted; mapWrite buffers go to the host-visible type.
func vkM1p(sizes []int64, mapRead, mapWrite []bool) int64 {
	dev := vkPacker{first: vkDevFirst, max: vkDevMax}
	host := vkPacker{first: vkHostFirst, max: vkHostMax}
	dev.place(vkInternal)
	for i, n := range sizes {
		switch {
		case mapRead != nil && mapRead[i]:
		case mapWrite != nil && mapWrite[i]:
			host.place(vkAligned(n))
		default:
			dev.place(vkAligned(n))
		}
	}
	return dev.total + max(host.total, vkHostKept) + vkDeviceConst
}

// vkPlanBuffers is the buffer list a WebGPU build of m makes, from the model alone: the planner's own list
// (decoder.Model.WebGPUDeviceBuffersForTest), so the census and the plan price the same buffers. slotKV is one further slot's K and V.
func vkPlanBuffers(m *decoder.Model, ctx int, kvF16, kvI8 bool) (slot0, slotKV []int64) {
	return m.WebGPUDeviceBuffersForTest(ctx, kvF16, kvI8)
}

// vkSlots is newKVSlot's outcome as M1′ predicts it on a device that can hold capacity bytes of Vulkan blocks: slot 0 is placed,
// then each further slot's K and V buffers and the headroom probe, which takes a block of its own and is freed; a slot is granted
// while the blocks and the probe fit. use is M1′'s device use with slot 0 alone.
func vkSlots(want int, slot0, slotKV []int64, capacity int64) (granted int, use int64) {
	dev := vkPacker{first: vkDevFirst, max: vkDevMax}
	dev.place(vkInternal)
	for _, n := range slot0 {
		dev.place(vkAligned(n))
	}
	use = dev.total + vkHostKept + vkDeviceConst
	headroom := int64(webgpuSlotHeadroom)
	if slotHeadroomPlantForTest > 0 {
		headroom = slotHeadroomPlantForTest
	}
	granted = 1
	for s := 1; s < want; s++ {
		next := dev.clone()
		for _, n := range slotKV {
			next.place(vkAligned(n))
		}
		if next.total+vkAligned(headroom) > capacity {
			break
		}
		dev = next
		granted++
	}
	return granted, use
}

// TestVkM1p_matchesTheScorer is G-VR0: on G-VK1's six recorded buffer lists, the census's packer and the planner's both give
// score_vk1.py's m1p_predict to the byte.
func TestVkM1p_matchesTheScorer(t *testing.T) {
	want := map[string]int64{ // python3 score_vk1.py's m1p_predict(bufs, True) on each pass1-<tag>-probe-buffers.tsv
		"llama1b": 2231369728, "mistral7b": 6794772480, "qwen0.5b": 1157627904,
		"qwen1.5b-ctx4096": 1694498816, "qwen1.5b": 2231369728, "qwen7b": 6802898944,
	}
	dir := "../docs/measurements/sb2-vk-2026-10-31"
	for tag, w := range want {
		b, err := os.ReadFile(filepath.Join(dir, "pass1-"+tag+"-probe-buffers.tsv"))
		if err != nil {
			t.Fatal(err)
		}
		var sizes []int64
		var mr, mw []bool
		for _, row := range strings.Split(strings.TrimSpace(string(b)), "\n")[1:] {
			f := strings.Split(row, "\t")
			n, err1 := strconv.ParseInt(f[1], 10, 64)
			u, err2 := strconv.ParseUint(strings.TrimPrefix(f[2], "0x"), 16, 64)
			if err1 != nil || err2 != nil {
				t.Fatalf("%s: %q: %v %v", tag, row, err1, err2)
			}
			sizes, mr, mw = append(sizes, n), append(mr, u&1 != 0), append(mw, u&2 != 0)
		}
		if got := vkM1p(sizes, mr, mw); got != w {
			t.Errorf("%s: Go M1′ %d, the scorer %d (%+d)", tag, got, w, got-w)
		}
		var dev []int64 // the planner's packer takes the device-local buffers; these lists hold no host-visible upload buffer
		for i, n := range sizes {
			if mw[i] {
				t.Fatalf("%s: a host-visible upload buffer in the list: the planner's packer does not take one", tag)
			}
			if !mr[i] {
				dev = append(dev, n)
			}
		}
		if got := decoder.WebGPUVkBlocksForTest(dev) + decoder.WebGPUVkFixed; got != w {
			t.Errorf("%s: the planner's M1′ %d, the scorer %d (%+d)", tag, got, w, got-w)
		}
	}
}

// TestBackendSlotCensus_webgpu_m1p is G-VR2's restatement and plants: every discrete line on a profile with a measured CUDA context
// carries M1′'s figures; a capacity 1 GiB lower changes discrete lines only; a capacity one byte short of slot 0 and the probe turns
// every discrete line that granted more than one slot to one, and changes no unified line.
func TestBackendSlotCensus_webgpu_m1p(t *testing.T) {
	clean, _ := buildWebGPUSlotCensus(t)
	restated, multi := 0, 0
	for _, l := range clean {
		f := strings.Fields(l)
		if len(f) < 6 || f[5] != "discrete" {
			continue
		}
		if _, ok := vkCUDAContext[f[4]]; ok {
			restated++
			if !strings.Contains(l, " capacity=") || !strings.Contains(l, "(M1′)") {
				t.Errorf("a discrete line on a profile with a measured CUDA context is not restated: %s", l)
			}
			if !strings.Contains(l, " slots=1/") && !strings.Contains(l, " slots=1~/") {
				multi++
			}
		}
	}
	if restated == 0 || multi == 0 {
		t.Fatalf("%d restated discrete lines, %d granting more than one slot: nothing for the plants to change", restated, multi)
	}
	plant := func(name string, set func(), undo func()) []string {
		set()
		defer undo()
		p, _ := buildWebGPUSlotCensus(t)
		if len(p) != len(clean) {
			t.Fatalf("%s: %d lines, want %d", name, len(p), len(clean))
		}
		for i := range clean {
			if !strings.Contains(clean[i], " discrete ") && p[i] != clean[i] {
				t.Errorf("%s: a unified line changed:\n%s\n%s", name, clean[i], p[i])
			}
		}
		return p
	}
	lower := plant("capacity-1GiB", func() { vkCapacityPlantForTest = 1 << 30 }, func() { vkCapacityPlantForTest = 0 })
	changed := 0
	for i := range clean {
		if lower[i] != clean[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Errorf("capacity-1GiB: no discrete line changed")
	}
	tight := plant("tight", func() { vkCapacityTightForTest = true }, func() { vkCapacityTightForTest = false })
	turned := 0
	for i := range clean {
		if !strings.Contains(clean[i], " capacity=") || strings.Contains(clean[i], " slots=1/") || strings.Contains(clean[i], " slots=1~/") {
			continue
		}
		if !strings.Contains(tight[i], " slots=1/") && !strings.Contains(tight[i], " slots=1~/") {
			t.Errorf("tight: a restated line that granted more than one does not grant one: %s", tight[i])
			continue
		}
		turned++
	}
	t.Logf("%d discrete lines restated (%d grant more than one slot); capacity-1GiB changed %d; tight turned %d to one slot", restated, multi, changed, turned)
}
