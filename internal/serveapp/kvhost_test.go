package serveapp

import (
	"strings"
	"testing"
	"time"
)

// -kv-host-bytes: the flag's grammar, its default (a fraction of RAM under a cap) and what the startup banner and the request log say.
func TestKVHostBytes_parseAndResolve(t *testing.T) {
	for _, tc := range []struct {
		in   string
		n    int64
		auto bool
		bad  bool
	}{
		{"", 0, true, false}, {"auto", 0, true, false}, {"AUTO", 0, true, false}, {"0", 0, false, false}, {"123", 123, false, false},
		{"1k", 1 << 10, false, false}, {"2G", 2 << 30, false, false}, {"512MiB", 512 << 20, false, false}, {"1gib", 1 << 30, false, false},
		{"-1", 0, false, true}, {"x", 0, false, true}, {"1.5G", 0, false, true}, {"9999999999999999G", 0, false, true}, {"G", 0, false, true},
	} {
		n, auto, err := parseKVHostBytes(tc.in)
		if (err != nil) != tc.bad || (err == nil && (n != tc.n || auto != tc.auto)) {
			t.Errorf("parse %q: %d, %v, %v; want %d, %v, bad=%v", tc.in, n, auto, err, tc.n, tc.auto, tc.bad)
		}
	}
	const g = int64(1) << 30
	for _, tc := range []struct {
		flag string
		ram  int64
		want int64
		how  string
	}{
		{"auto", 64 * g, 64 * g / 10, "auto: 1/10 of 64.0 GiB"},
		{"auto", 128 * g, 8 * g, "at most 8.0 GiB"}, // a tenth of this much RAM is over the cap
		{"auto", 16 * g, 16 * g / 10, "1/10 of 16.0 GiB"},
		{"auto", 0, 0, "unknown"},
		{"0", 64 * g, 0, "-kv-host-bytes 0"},
		{"3G", 64 * g, 3 * g, "-kv-host-bytes 3G"},
		{"3G", 0, 3 * g, "-kv-host-bytes 3G"}, // a stated budget does not depend on knowing the RAM
	} {
		got, how, err := resolveKVHostBytes(tc.flag, tc.ram)
		if err != nil || got != tc.want || !strings.Contains(how, tc.how) {
			t.Errorf("resolve %q with %d bytes of RAM: %d %q (%v); want %d containing %q", tc.flag, tc.ram, got, how, err, tc.want, tc.how)
		}
	}
	if _, _, err := resolveKVHostBytes("nope", g); err == nil {
		t.Error("a bad value resolved")
	}
	if err := (config{kvHostBytesFlag: "1x"}).validate(); err == nil || !strings.Contains(err.Error(), "kv-host-bytes") {
		t.Errorf("validate accepted a bad -kv-host-bytes: %v", err)
	}
}

func TestKVHostBytes_bannerAndLog(t *testing.T) {
	line := func(f bannerFacts, cfg config) string {
		for _, l := range modelBannerFrom(f, cfg) {
			if strings.HasPrefix(l, "prefix snapshots:") {
				return l
			}
		}
		return ""
	}
	on := config{kvHostBytes: 6 << 30, kvHostBytesHow: "auto: 1/10 of 64.0 GiB of RAM, at most 8.0 GiB"}
	if l := line(bannerFacts{resident: true, snapshots: true}, on); !strings.Contains(l, "budget 6.0 GiB") || !strings.Contains(l, "0 B held") || !strings.Contains(l, "pinned only while a snapshot is held") {
		t.Errorf("on: %q", l)
	}
	if l := line(bannerFacts{resident: true, snapshots: false}, on); !strings.Contains(l, "not on this backend") {
		t.Errorf("a resident that cannot: %q", l)
	}
	if l := line(bannerFacts{resident: true}, config{kvHostBytesHow: "-kv-host-bytes 0"}); l != "prefix snapshots: OFF (-kv-host-bytes 0)" {
		t.Errorf("off: %q", l)
	}
	if l := line(bannerFacts{resident: false}, on); l != "" {
		t.Errorf("a model off the resident path reports snapshots: %q", l)
	}
	// the request log names the rows a restore brought back, and only then
	tr := &reqTrace{start: time.Now(), model: "m"}
	if l := tr.line("/v1/chat/completions", 200, time.Now()); strings.Contains(l, "snap=") {
		t.Errorf("a request with no restore: %q", l)
	}
	tr.snapRows = 7488
	if l := tr.line("/v1/chat/completions", 200, time.Now()); !strings.HasSuffix(l, " snap=7488") {
		t.Errorf("a restored request: %q", l)
	}
}
