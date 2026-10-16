package serveapp

import (
	"fmt"
	"strconv"
	"strings"
)

// -kv-host-bytes: the host memory prefix snapshots may hold (decoder.Options.KVHostBytes, docs/tasks/task-kv7-prefix-snapshot-2026-10.md).
// The default is a tenth of the machine's physical RAM, at most 8 GiB; pinned memory is allocated when a snapshot is taken and
// freed when it is dropped, so a server that never takes one holds nothing.

const (
	kvHostFraction = 10      // the default budget is 1/kvHostFraction of physical RAM
	kvHostCap      = 8 << 30 // and at most this
)

// parseKVHostBytes reads the flag: "auto" (or empty) for the default, "0" for off, else a byte count with an optional K, M or G
// suffix (powers of 1024, an optional trailing "iB" or "B" accepted).
func parseKVHostBytes(s string) (n int64, auto bool, err error) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "auto") {
		return 0, true, nil
	}
	t := strings.TrimSuffix(strings.TrimSuffix(strings.ToLower(s), "b"), "i")
	mult := int64(1)
	if t != "" {
		switch t[len(t)-1] {
		case 'k':
			mult, t = 1<<10, t[:len(t)-1]
		case 'm':
			mult, t = 1<<20, t[:len(t)-1]
		case 'g':
			mult, t = 1<<30, t[:len(t)-1]
		}
	}
	v, perr := strconv.ParseInt(t, 10, 64)
	if perr != nil || v < 0 || (mult > 1 && v > (1<<62)/mult) {
		return 0, false, fmt.Errorf("-kv-host-bytes wants auto, 0, or a byte count with an optional K, M or G (got %q)", s)
	}
	return v * mult, false, nil
}

// resolveKVHostBytes is the budget in bytes for the flag's value and the machine's physical RAM (0 when unknown), and how it was
// reached, for the startup line.
func resolveKVHostBytes(flag string, ram int64) (int64, string, error) {
	n, auto, err := parseKVHostBytes(flag)
	if err != nil {
		return 0, "", err
	}
	switch {
	case !auto && n == 0:
		return 0, "off: -kv-host-bytes 0", nil
	case !auto:
		return n, "-kv-host-bytes " + strings.TrimSpace(flag), nil
	case ram <= 0:
		return 0, "off: auto needs the machine's physical RAM, which is unknown here", nil
	}
	b := min(ram/kvHostFraction, kvHostCap)
	return b, fmt.Sprintf("auto: 1/%d of %s of RAM, at most %s", kvHostFraction, gib(ram), gib(kvHostCap)), nil
}

// gib is n bytes as GiB with one decimal ("62.7 GiB"), or MiB below a GiB.
func gib(n int64) string {
	if n >= 1<<30 {
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	}
	return fmt.Sprintf("%.0f MiB", float64(n)/(1<<20))
}
