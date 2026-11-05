package serveapp

import "testing"

// TestAddrIsLoopback covers the cases the startup auth/TLS-warning checks rely on:
// loopback (127.0.0.1, localhost, ::1) never requires -api-key; every-interface
// binds (0.0.0.0, [::], a bare port) and any other hostname always do, since they
// may be reachable from another machine on the network.
func TestAddrIsLoopback(t *testing.T) {
	cases := []struct {
		addr string
		want bool
	}{
		{"127.0.0.1:8080", true},
		{"localhost:8080", true},
		{"[::1]:8080", true},
		{"0.0.0.0:8080", false},
		{"[::]:8080", false},
		{":8080", false},            // bare port: binds every interface
		{"192.168.1.5:8080", false}, // a real LAN IP
		{"example.com:8080", false}, // a hostname might resolve off-box
		{"127.0.0.1", true},         // no port
	}
	for _, c := range cases {
		if got := addrIsLoopback(c.addr); got != c.want {
			t.Errorf("addrIsLoopback(%q) = %v, want %v", c.addr, got, c.want)
		}
	}
}
