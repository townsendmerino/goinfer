package mem

import "testing"

func TestGiB_isBinaryWithTwoDecimals(t *testing.T) {
	for _, c := range []struct {
		b    int64
		want string
	}{{0, "0.00 GiB"}, {1 << 30, "1.00 GiB"}, {8_000_000_000, "7.45 GiB"}, {3 << 29, "1.50 GiB"}} {
		if got := GiB(c.b); got != c.want {
			t.Errorf("GiB(%d) = %q, want %q", c.b, got, c.want)
		}
	}
}
