//go:build darwin

package metal

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
)

// TestMC3_kvSlotFootprintProbe is T1.6 (E-P09, docs/audit-metal-2026-09-30.md), a probe rather than a gate: this
// process's footprint, and the IOAccelerator (GPU buffer) share of it, with the GOINFER_METAL_MC3 checkpoint resident
// at 1, 4 and again 1 KV slots, read right after the build and again after one token. It shows whether the extra
// slots' KV pages count before anything writes them; CurrentAllocatedSize, which counts every buffer whatever its
// pages, is beside it. The second 1-slot arm is the control. Run each arm in its own process, or the fit guard refuses
// the third load on a 16 GB Mac (the first two have not handed all their memory back). No assertion: the numbers go to
// the record.
//
//	for a in arm0_slots1 arm1_slots4 arm2_slots1; do GOINFER_METAL_MC3=1 go test -count=1 -run "^TestMC3_kvSlotFootprintProbe$/^$a$" -v ./metal/; done
func TestMC3_kvSlotFootprintProbe(t *testing.T) {
	for i, slots := range []int{1, 4, 1} {
		t.Run(fmt.Sprintf("arm%d_slots%d", i, slots), func(t *testing.T) {
			_, r := mc3LoadCheckpoint(t, slots, 4096)
			runtime.GC()
			debug.FreeOSMemory()
			b, bIO, bCat := procFootprint(t)
			bDev := r.d.CurrentAllocatedSize()
			r.ForwardEmb(mc3Emb(r, 1), 0)
			a, aIO, aCat := procFootprint(t)
			kv := 0
			for l := range r.layers {
				kv += 2 * 2 * r.layers[l].geom.kvDim * r.ctxCap // K and V, f16
			}
			line := fmt.Sprintf("[kv-footprint] %d slot(s) (%d built, %.0f MB KV each): after build footprint %.0f MB, IOAccelerator %.0f MB, "+
				"device allocated %.0f MB; after one token footprint %.0f MB, IOAccelerator %.0f MB\n",
				slots, len(r.kvSlotBufs), float64(kv)/(1<<20), footprintMB(b), footprintMB(bIO), float64(bDev)/(1<<20), footprintMB(a), footprintMB(aIO))
			line += fmt.Sprintf("[kv-footprint]   categories of 16 MB or more after build: %s\n[kv-footprint]   after one token: %s\n", bCat, aCat)
			fmt.Fprint(os.Stderr, line)
			t.Log(strings.TrimSpace(line))
		})
	}
}

var (
	footprintTotalRe = regexp.MustCompile(`Footprint: (\d+) B`)
	footprintRowRe   = regexp.MustCompile(`(?m)^\s*(\d+) B\s+\d+ B\s+\d+ B\s+\d+\s+(\S.*?)\s*$`)
)

// procFootprint is footprint(1)'s total for this process, the dirty bytes of its IOAccelerator rows, and every
// category holding 16 MB or more of dirty memory.
func procFootprint(t *testing.T) (total, ioaccel int64, big string) {
	t.Helper()
	out, err := exec.Command("footprint", "-f", "bytes", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Skipf("footprint: %v", err)
	}
	m := footprintTotalRe.FindSubmatch(out)
	if m == nil {
		t.Fatalf("footprint printed no total:\n%s", out)
	}
	total, _ = strconv.ParseInt(string(m[1]), 10, 64)
	var cats []string
	for _, row := range footprintRowRe.FindAllSubmatch(out, -1) {
		n, _ := strconv.ParseInt(string(row[1]), 10, 64)
		name := string(row[2])
		if strings.HasPrefix(name, "IOAccelerator") {
			ioaccel += n // footprint can list more than one IOAccelerator row
		}
		if n >= 16<<20 && name != "TOTAL" {
			cats = append(cats, fmt.Sprintf("%s %.0f", name, footprintMB(n)))
		}
	}
	return total, ioaccel, strings.Join(cats, ", ")
}

func footprintMB(b int64) float64 { return float64(b) / (1 << 20) }
