package swapguard

import (
	"strings"
	"testing"
)

func ok(v int64) func() (int64, bool) { return func() (int64, bool) { return v, true } }

const mb = int64(1024 * 1024)

func TestSelect(t *testing.T) {
	sys := ok(0)
	unreadable := func() (int64, bool) { return 0, false }
	for _, c := range []struct {
		name, env string
		own       func() (int64, bool)
		why       string
		want      []string // "name:thresholdMB"
		banner    string
	}{
		{"per-process reader: own at the default, system as the backstop", "", ok(0), "", []string{"server swap:512", "system swap:4096"}, "backstop"},
		{"an explicit threshold moves the own trip", "100", ok(0), "", []string{"server swap:100", "system swap:4096"}, "+100 MB"},
		{"a threshold above the backstop raises the backstop with it", "8192", ok(0), "", []string{"server swap:8192", "system swap:8192"}, "+8192 MB as a backstop"},
		{"no per-process reader: system-wide at the default, and the banner says why", "", nil, "macOS has no per-process swap figure", []string{"system swap:512"}, "macOS has no per-process swap figure"},
		{"VmSwap unreadable: system-wide at the default, and the banner says so", "", unreadable, "", []string{"system swap:512"}, "VmSwap is not readable"},
		{"fallback honours the threshold", "700", nil, "no reader", []string{"system swap:700"}, "+700 MB"},
	} {
		t.Setenv("GOINFER_SWAP_GUARD", c.env)
		srcs, banner, on := Select(c.own, c.why, sys)
		if !on {
			t.Errorf("%s: guard reported off", c.name)
			continue
		}
		var got []string
		for _, s := range srcs {
			got = append(got, s.Name+":"+itoa(s.Threshold/mb))
		}
		if strings.Join(got, ",") != strings.Join(c.want, ",") || !strings.Contains(banner, c.banner) {
			t.Errorf("%s: sources %v, banner %q; want %v mentioning %q", c.name, got, banner, c.want, c.banner)
		}
	}
	t.Setenv("GOINFER_SWAP_GUARD", "off")
	if srcs, _, on := Select(ok(0), "", sys); on || srcs != nil {
		t.Errorf("off: %v, %v", srcs, on)
	}
}

func itoa(v int64) string {
	if v == 0 {
		return "0"
	}
	var b []byte
	for ; v > 0; v /= 10 {
		b = append([]byte{byte('0' + v%10)}, b...)
	}
	return string(b)
}
